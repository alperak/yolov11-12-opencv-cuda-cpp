### Real Time Object Detection using YOLOv11/12, OpenCV with CUDA backend in C++

![C++](https://img.shields.io/badge/C++-00599C?style=for-the-badge&logo=c%2B%2B)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv)
![CUDA](https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia)
![ONNX](https://img.shields.io/badge/ONNX-005CED?style=for-the-badge&logo=onnx)
![YOLOv11](https://img.shields.io/badge/YOLOv11-00FFFF?style=for-the-badge&logo=yolo)

### Project overview

This implementation is designed as a demo to test COCO trained YOLOv11/12 detection models.

This project implements YOLOv11/12 object detection in C++ using OpenCV with CUDA backend. It supports real time inference on NVIDIA GPUs and uses an ONNX model (e.g, yolov11s.onnx, yolov12s.onnx).

For simplicity, it runs entirely on the main thread, so separate threads for camera capture, inference and drawing aren't used.

The focus is on clarity and educational value.

### Demo

A demo captured using **YOLOV11x**:

![demo](https://github.com/user-attachments/assets/c699fb75-44f8-403e-9a26-0f46b1a9d10f)

### Tested Environment

This project has been successfully tested with the:

- Ubuntu: `24.04.2 LTS`
- OpenCV and OpenCV Contrib: `4.12.0`
- CUDA: `12.9.1`
- cuDNN: `8.9.7`
- CMake: `3.10+`

### Quick Start

(Note: The code doesn't currently get configuration input from the user so need to update code and build if you need different configurations. It would be nice if someone does :) )

Before building, you may need to update the following lines to suit your configuration:

```cpp
// Initialize the detector with model path, model input size, labels text path, thresholds
// and inference target(GPU/CPU)

Inference detector("../model/yolo11s.onnx", cv::Size(640, 640), "../model/labels.txt",
                    { .modelScoreThreshold = 0.45f, .modelNMSThreshold = 0.50f },
                    InferenceTarget::GPU);

// Open camera device
// Your camera device id may be different so use 'v4l2-ctl --list-devices' on terminal
// and update it to 1 or what ever available for you.
cv::VideoCapture cap(0);
```

After the configurations update, you can follow to the build step:

```bash
# Fetch the project
git clone https://github.com/alperak/yolov11-12-opencv-cuda-cpp.git
cd yolov11-12-opencv-cuda-cpp

# Create build directory
mkdir build && cd build

# Configure with CMake (If you want documentation, enable -DBUILD_DOCS=ON and Doxygen must be installed.)
cmake ..

# Build the project
make -j$(nproc)

# Run the executable
./yolo-inference
```

### Download YOLOv11/12 Models

If you want to try models other than the YOLOv11s in the project, download pretrained YOLOv11/12 models from [Ultralytics](https://docs.ultralytics.com/tasks/detect/#models):

| Model | Size<br>(pixels) | mAP<sup>val</sup><br>50-95 | Speed<br>CPU ONNX<br>(ms) | Speed<br>T4 TensorRT10<br>(ms) | Params<br>(M) | FLOPs<br>(B) |
|---------|:----------------:|:---------------------------:|:--------------------------:|:-------------------------------:|:-------------:|:------------:|
| [YOLOv11n](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt) | 640 | 39.5 | 56.1 ± 0.8 | 1.5 ± 0.0 | 2.6 | 6.5 |
| [YOLOv11s](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt) | 640 | 47.0 | 90.0 ± 1.2 | 2.5 ± 0.0 | 9.4 | 21.5 |
| [YOLOv11m](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt) | 640 | 51.5 | 183.2 ± 2.0 | 4.7 ± 0.1 | 20.1 | 68.0 |
| [YOLOv11l](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt) | 640 | 53.4 | 238.6 ± 1.4 | 6.2 ± 0.1 | 25.3 | 86.9 |
| [YOLOv11x](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt) | 640 | 54.7 | 462.8 ± 6.7 | 11.3 ± 0.2 | 56.9 | 194.9 |


| Model | Size<br>(pixels) | mAP<sup>val</sup><br>50-95 | Speed<br>CPU ONNX<br>(ms) | Speed<br>T4 TensorRT<br>(ms) | Params<br>(M) | FLOPs<br>(B) | Comparison<br>(mAP/Speed) |
|---------|:----------------:|:---------------------------:|:--------------------------:|:-----------------------------:|:-------------:|:------------:|:------------------------:|
| [YOLO12n](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12n.pt) | 640 | 40.6 | - | 1.64 | 2.6 | 6.5 | +2.1%/-9%<br>(vs. YOLOv10n) |
| [YOLO12s](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12s.pt) | 640 | 48.0 | - | 2.61 | 9.3 | 21.4 | +0.1%/+42%<br>(vs. RT-DETRv2) |
| [YOLO12m](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12m.pt) | 640 | 52.5 | - | 4.86 | 20.2 | 67.5 | +1.0%/-3%<br>(vs. YOLOv11m) |
| [YOLO12l](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12l.pt) | 640 | 53.7 | - | 6.77 | 26.4 | 88.9 | +0.4%/-8%<br>(vs. YOLOv11l) |
| [YOLO12x](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12x.pt) | 640 | 55.2 | - | 11.79 | 59.1 | 199.0 | +0.6%/-4%<br>(vs. YOLOv11x) |

### Convert PyTorch Models to ONNX

`YOLOv11s` already converted and available in the project but if you want to use another models, you can convert by following the steps:

```bash
# Install Ultralytics package
pip install ultralytics

# For example, we want to try `YOLOv11x` so need to download the `YOLOv11x.pt` into our model directory
# and convert model to `ONNX` format like this:

python convert_pt_to_onnx_model.py yolo11x.pt 
```

You should see this output when you try YOLOv11x:

```
python convert_pt_to_onnx_model.py yolo11x.pt
Ultralytics 8.3.202 🚀 Python-3.12.3 torch-2.8.0+cu128 CPU (Intel Core i7-8700K 3.70GHz)
YOLO11x summary (fused): 190 layers, 56,919,424 parameters, 0 gradients, 194.9 GFLOPs

PyTorch: starting from 'yolo11x.pt' with input shape (1, 3, 640, 640) BCHW and output shape(s) (1, 84, 8400) (109.3 MB)

ONNX: starting export with onnx 1.19.0 opset 19...
ONNX: slimming with onnxslim 0.1.68...
ONNX: export success ✅ 3.7s, saved as 'yolo11x.onnx' (217.5 MB)

Export complete (5.5s)
Results saved to /home/alper/cpp-projects/YOLOv11-OpenCV-CUDA-Cpp/model
Predict:         yolo predict task=detect model=yolo11x.onnx imgsz=640  
Validate:        yolo val task=detect model=yolo11x.onnx imgsz=640 data=/ultralytics/ultralytics/cfg/datasets/coco.yaml  
Visualize:       https://netron.app
```
