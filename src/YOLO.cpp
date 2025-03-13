#include "YOLO.h"
#include "logging.h"
#include "cuda_utils.h"
#include "macros.h"
#include "preprocess.h"
#include "postprocess.h"
#include <NvOnnxParser.h>
#include "common.h"
#include <fstream>
#include <iostream>


static Logger logger;
#define isFP16 true
#define warmup true


YOLO::YOLO(const std::string &model_path, nvinfer1::ILogger& logger){
    if (!loadEngine(model_path, logger)) {
        build(model_path, logger);
        saveEngine(model_path);
    }

    // Create RAII buffer manager object
    buffers = std::make_shared<BufferManager>(engine, 1);

    // Register the input and output buffers
    for (int32_t i = 0, e = engine->getNbIOTensors(); i < e; i++)
    {
        auto const name = engine->getIOTensorName(i);
        context->setTensorAddress(name, buffers->getDeviceBuffer(name));
    }

    #if NV_TENSORRT_MAJOR < 8
        input_h = engine->getBindingDimensions(0).d[2];
        input_w = engine->getBindingDimensions(0).d[3];
        detection_attribute_size = engine->getBindingDimensions(1).d[1];
        num_detections = engine->getBindingDimensions(1).d[2];
    #else
        auto input_name = engine->getIOTensorName(0);
        auto output_name = engine->getIOTensorName(1);

        auto input_dims = engine->getTensorShape(input_name);
        auto output_dims = engine->getTensorShape(output_name);

        input_h = input_dims.d[2];
        input_w = input_dims.d[3];
        detection_attribute_size = output_dims.d[1];
        num_detections = output_dims.d[2];
    #endif
        num_classes = detection_attribute_size - 4;

    cuda_preprocess_init(MAX_IMAGE_SIZE);

    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMalloc(&transposeDevice, detection_attribute_size * num_detections * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&decodeDevice, (1 + modelOutputRows * modelOutputDims) * sizeof(float)));

    if (warmup) {
        for (int i = 0; i < 10; i++) {
            this->infer();
        }
        // printf("model warmup 10 times\n");
    }
}

YOLO::~YOLO(){
    // Release stream and buffers
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(transposeDevice));
    CUDA_CHECK(cudaFree(decodeDevice));
    // Destroy the engine
    cuda_preprocess_destroy();
}

void YOLO::preprocess(cv::Mat& image) {
    // Preprocessing data on gpu
    void *deviceDataBuffer = buffers->getDeviceBuffer(engine->getIOTensorName(0));
    cuda_preprocess(image.ptr(), image.cols, image.rows, (float*)deviceDataBuffer, input_w, input_h, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void YOLO::infer(){
#if NV_TENSORRT_MAJOR < 10
    context->enqueueV2(buffers->getDeviceBindings().data(), stream, nullptr);
#else
    context->executeV2(buffers->getDeviceBindings().data());
#endif
}

void YOLO::postprocess(std::vector<Detection>& output){
    float* deviceDataBuffer = (float*)buffers->getDeviceBuffer(engine->getIOTensorName(1));
    // transpose [1 84 8400] convert to [1 8400 84]
    transpose(deviceDataBuffer, transposeDevice, num_detections, num_classes + 4, stream);
    // convert [1 8400 84] to [1 7001]
    decode(transposeDevice, decodeDevice, num_detections, num_classes, conf_threshold, modelOutputRows, modelOutputDims, stream);
    // cuda nms
    nms(decodeDevice, nms_threshold, modelOutputRows, modelOutputDims, stream);
    // Memcpy from device output buffer to host output buffer
    float* cpu_output_buffer = (float*)buffers->getHostBuffer(engine->getIOTensorName(1));
    CUDA_CHECK(cudaMemcpyAsync(cpu_output_buffer, decodeDevice, (1 + modelOutputRows * modelOutputDims) * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    int count = std::min((int)cpu_output_buffer[0], modelOutputRows); // specifying modelOutputRows as the upper limit
    for (int i = 0; i < count; i++){ // iterate up to count instead of nms_result size
        int pos = 1 + i * modelOutputDims;
        int keepFlag = (int)cpu_output_buffer[pos + 6];
        if (keepFlag == 0){
            continue;
        }
        Detection result;
        result.class_id = (int)cpu_output_buffer[pos + 5];
        result.conf = cpu_output_buffer[pos + 4];
        result.bbox.x = cpu_output_buffer[pos + 0];
        result.bbox.y = cpu_output_buffer[pos + 1];
        result.bbox.width = cpu_output_buffer[pos + 2] - cpu_output_buffer[pos + 0];
        result.bbox.height = cpu_output_buffer[pos + 3] - cpu_output_buffer[pos + 1];
        output.push_back(result);
    }
}

void YOLO::build(const std::string &onnxPath, nvinfer1::ILogger& logger){
    logger.log(nvinfer1::ILogger::Severity::kINFO, "Building TensorRT Engine from ONNX file");
    auto builder = nvinfer1::createInferBuilder(logger);
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    
    if (isFP16){
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }

    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);
    bool parsed = parser->parseFromFile(onnxPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO));
    if (!parsed){
        std::cerr << "Failed to parse onnx file: " << onnxPath << std::endl;
        return;
    }
    nvinfer1::IHostMemory* plan{ builder->buildSerializedNetwork(*network, *config) };

    // TensorRT runtime for deserializing the engine from file
    std::unique_ptr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(logger)};

    engine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan->data(), plan->size()));

    context = std::unique_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());

    delete network;
    delete config;
    delete parser;
    delete plan;
}

bool YOLO::saveEngine(const std::string& onnxpath){
    // Create an engine path from onnx path
    std::string engine_path;
    size_t dotIndex = onnxpath.find_last_of(".");
    if (dotIndex != std::string::npos){
        engine_path = onnxpath.substr(0, dotIndex) + ".engine";
    }
    else{
        return false;
    }

    // Save the engine to the path
    if (engine){
        nvinfer1::IHostMemory* data = engine->serialize();
        std::ofstream file;
        file.open(engine_path, std::ios::binary | std::ios::out);
        if (!file.is_open()){
            std::cout << "Create engine file" << engine_path << " failed" << std::endl;
            return 0;
        }
        file.write((const char*)data->data(), data->size());
        file.close();

        delete data;
    }
    return true;
}

bool YOLO::loadEngine(const std::string& filename, nvinfer1::ILogger& logger){
    // Create an engine path from onnx path
    std::string engine_path;
    size_t dotIndex = filename.find_last_of(".");
    if (dotIndex != std::string::npos){
        engine_path = filename.substr(0, dotIndex) + ".engine";
    }
    else{
        return false;
    }
    // Read the engine file
    std::ifstream engineStream(engine_path, std::ios::binary);
    if (!engineStream.good()) {
        logger.log(nvinfer1::ILogger::Severity::kERROR, "Failed to open engine file");
        return false;
    }
    engineStream.seekg(0, std::ios::end);
    const size_t modelSize = engineStream.tellg();
    engineStream.seekg(0, std::ios::beg);
    std::unique_ptr<char[]> engineData(new char[modelSize]);
    engineStream.read(engineData.get(), modelSize);
    engineStream.close();

    // Deserialize the tensorrt engine
    // TensorRT runtime for deserializing the engine from file
    std::unique_ptr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(logger)};
    engine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(engineData.get(), modelSize));
    context = std::unique_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
    return true;
}

void YOLO::draw(cv::Mat& image, const std::vector<Detection>& output){
    const float ratio_h = input_h / (float)image.rows;
    const float ratio_w = input_w / (float)image.cols;

    for (int i = 0; i < output.size(); i++){
        auto detection = output[i];
        auto box = detection.bbox;
        auto class_id = detection.class_id;
        auto conf = detection.conf;
        cv::Scalar color = cv::Scalar(COLORS[class_id][0], COLORS[class_id][1], COLORS[class_id][2]);

        if (ratio_h > ratio_w){
            box.x = box.x / ratio_w;
            box.y = (box.y - (input_h - ratio_w * image.rows) / 2) / ratio_w;
            box.width = box.width / ratio_w;
            box.height = box.height / ratio_w;
        }
        else{
            box.x = (box.x - (input_w - ratio_h * image.cols) / 2) / ratio_h;
            box.y = box.y / ratio_h;
            box.width = box.width / ratio_h;
            box.height = box.height / ratio_h;
        }
    
        rectangle(image, cv::Point(box.x, box.y), cv::Point(box.x + box.width, box.y + box.height), color, 3);

        // Detection box text
        std::string class_string = CLASS_NAMES[class_id] + " " + std::to_string(conf).substr(0, 4);
        cv::Size text_size = cv::getTextSize(class_string, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
        cv::Rect text_rect(box.x, box.y - 40, text_size.width + 10, text_size.height + 20);
        cv::rectangle(image, text_rect, color, cv::FILLED);
        cv::putText(image, class_string, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
    }
}