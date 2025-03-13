#pragma once  

#include "NvInfer.h"    // TensorRT library for high-performance inference
#include "buffers.h"     // Helper class to manage data buffers
#include <opencv2/opencv.hpp>  // OpenCV for image processing

// Struct to store detection results
struct Detection {
    float conf;  // Confidence score of the detection
    int class_id;  // Class ID of the detected object (e.g., person, car, etc.)
    cv::Rect bbox;  // Bounding box coordinates around the detected object
};

// Main class for the YOLO model
class YOLO {

public:
    // Constructor: Loads the TensorRT engine and initializes the model
    YOLO(const std::string &model_path, nvinfer1::ILogger& logger);

    // Destructor: Cleans up resources used by the model
    ~YOLO();

    // Preprocess the input image to match the model's input format
    void preprocess(cv::Mat& image);

    // Run inference on the preprocessed image
    void infer();

    // Postprocess the model's output to extract detection results
    void postprocess(std::vector<Detection>& output);

    // Draw bounding boxes and labels on the original image
    void draw(cv::Mat& image, const std::vector<Detection>& output);

private:
    // CUDA stream for asynchronous execution
    cudaStream_t stream;

    // TensorRT engine used to execute the network
    std::shared_ptr<nvinfer1::ICudaEngine> engine;

    // Execution context for running inference with the engine
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    std::shared_ptr<BufferManager> buffers;

    // Model parameters
    int input_w;  // Input image width expected by the model
    int input_h;  // Input image height expected by the model
    int num_detections;  // Number of detections output by the model
    int detection_attribute_size;  // Attributes (e.g., bbox, class) per detection
    int num_classes = 80;  // Number of classes (e.g., COCO dataset has 80 classes)

    // Maximum supported image size (used for memory allocation checks)
    const int MAX_IMAGE_SIZE = 4096 * 4096;

    const int modelOutputRows {300};
    const int modelOutputDims {7};
    float *transposeDevice;
    float *decodeDevice;

    // Confidence threshold for filtering detections
    float conf_threshold = 0.3f;

    // Non-Maximum Suppression (NMS) threshold to remove duplicate boxes
    float nms_threshold = 0.4f;

    // Colors for drawing bounding boxes for each class
    std::vector<cv::Scalar> colors;

    // Build TensorRT engine from an ONNX model file (if applicable)
    void build(const std::string& onnxPath, nvinfer1::ILogger& logger);

    // Save the built TensorRT engine to a file
    bool saveEngine(const std::string& filename);

    // Load the TensorRT engine from a file
    bool loadEngine(const std::string& filename, nvinfer1::ILogger& logger);
};
