#ifdef _WIN32
#include <windows.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <iostream>
#include <string>
#include "YOLO.h"
#include "performance.h"

bool IsPathExist(const std::string& path) {
#ifdef _WIN32
    DWORD fileAttributes = GetFileAttributesA(path.c_str());
    return (fileAttributes != INVALID_FILE_ATTRIBUTES);
#else
    return (access(path.c_str(), F_OK) == 0);
#endif
}

bool IsFile(const std::string& path) {
    if (!IsPathExist(path)) {
        printf("%s:%d %s not exist\n", __FILE__, __LINE__, path.c_str());
        return false;
    }

#ifdef _WIN32
    DWORD fileAttributes = GetFileAttributesA(path.c_str());
    return ((fileAttributes != INVALID_FILE_ATTRIBUTES) && ((fileAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0));
#else
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0 && S_ISREG(buffer.st_mode));
#endif
}

/**
 * @brief Setting up Tensorrt logger
*/
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        // Only output logs with severity greater than warning
        if (severity <= Severity::kINFO)
            std::cout << msg << std::endl;
    }
}logger;

int main(int argc, char** argv){

    const std::string engine_file_path{ argv[1] };
    const std::string path{ argv[2] };
    std::vector<std::string> imagePathList;
    bool isVideo{ false };

    assert(argc == 3);
    if (IsFile(path)){
        std::string suffix = path.substr(path.find_last_of('.') + 1);
        if (suffix == "jpg" || suffix == "jpeg" || suffix == "png"){
            imagePathList.push_back(path);
        }
        else if (suffix == "mp4" || suffix == "avi" || suffix == "m4v" || suffix == "mpeg" || suffix == "mov" || suffix == "mkv" || suffix == "webm"){
            isVideo = true;
        }
        else {
            printf("suffix %s is wrong !!!\n", suffix.c_str());
            abort();
        }
    }
    else if (IsPathExist(path)){
        cv::glob(path + "/*.jpg", imagePathList);
    }

    // init model
    YOLO model(engine_file_path, logger);


    if (isVideo) {
        std::cout << "Opening video: " << path << std::endl;
        cv::VideoCapture cap(path);

        if (!cap.isOpened()) {
            std::cerr << "Error: Cannot open video file!" << std::endl;
            return -1;
        }

        // Get frame width, height, and FPS
        int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        int fps = static_cast<int>(cap.get(cv::CAP_PROP_FPS));

        // Define the codec and create VideoWriter object
        cv::VideoWriter videoWriter("output.mp4",
                                    cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                                    fps,
                                    cv::Size(frameWidth, frameHeight));

        if (!videoWriter.isOpened()) {
            std::cerr << "Error: Cannot open VideoWriter!" << std::endl;
            return -1;
        }

        while (true) {
            cv::Mat image;
            cap >> image;  // Read a frame

            if (image.empty()) {
                break;  // Exit loop if no more frames
            }

            std::vector<Detection> objects;

            std::cout << "Image preprocess will start" << std::endl;
            model.preprocess(image);
            std::cout << "Image preprocess done" << std::endl;

            auto start = std::chrono::system_clock::now();
            model.infer();
            std::cout << "Model inference done" << std::endl;
            auto end = std::chrono::system_clock::now();

            model.postprocess(objects);
            model.draw(image, objects);
            std::cout << "Model postprocess done" << std::endl;

            auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
            printf("Cost %2.4lf ms\n", tc);

            // Write processed frame to output video
            videoWriter.write(image);

            if (cv::waitKey(1) == 27) {  // Press 'ESC' to exit early
                break;
            }
        }

        // Release resources
        cap.release();
        videoWriter.release();
        cv::destroyAllWindows();
    }


    else {
        // path to folder saves images
        for (const auto& imagePath : imagePathList){
            // open image
            cv::Mat image = cv::imread(imagePath);
            if (image.empty()){
                std::cerr << "Error reading image: " << imagePath << std::endl;
                continue;
            }

            std::vector<Detection> objects;
            {
                Perf p("Preprocess");
                model.preprocess(image);
            }

            {
                Perf p("Inference");
                model.infer();
            }

            {
                Perf p("Postprocess");
                model.postprocess(objects);
            }

            model.draw(image, objects);

            cv::imshow("Result", image);

            cv::waitKey(0);
        }
    }
    return 0;
}