﻿cmake_minimum_required(VERSION 3.12)
project(yolo)

find_package(CUDA REQUIRED)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
add_definitions(-DAPI_EXPORTS)

# Include CUDA directories
include_directories(${CUDA_INCLUDE_DIRS})

# Add source files
set(SOURCES
    main.cpp
    src/YOLO.cpp
)

# Add CUDA source files
set(CUDA_SOURCES src/preprocess.cu src/postprocess.cu)

# Add headers
include_directories(${PROJECT_SOURCE_DIR}/include)

# Find and include OpenCV
set(OpenCV_DIR "your OpenCV build directory path")

# Set TensorRT path if not set in environment variables
set(TENSORRT_DIR "your TensorRT directory path")

# Include OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Include TensorRT
include_directories(${TENSORRT_DIR}/include)
link_directories(${TENSORRT_DIR}/lib)
set(TENSORRT_LIBS nvinfer nvinfer_plugin nvonnxparser)

# Create an executable
cuda_add_executable(${PROJECT_NAME} ${SOURCES} ${CUDA_SOURCES})

# Link libraries
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    ${TENSORRT_LIBS}
)