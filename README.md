# YOLO-TensorRT
This project is an inference of YOLO (v11-v12) using TensorRT v10. Preprocess and postprocess are implemented via CUDA.
## Testing Environment
This project has been tested on CUDA v12.4 and TensorRT v10.9.
### Build
Fill correct paths in CMakeLists.txt.
```
cd PATH/TO/YOLO-TensorRT
mkdir build
cd build
cmake ..
make
```
### Run
```
./yolo PATH/TO/YOLO.onnx PATH/TO/IMAGE
```
This command will automatically generate TensorRT engine if it doesn't exist.