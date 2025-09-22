#include "Inference.hpp"

int main()
{
    try {
        // Check available CUDA devices
        std::cout << "CUDA Device(s) Number: "<< cv::cuda::getCudaEnabledDeviceCount() << '\n';
        cv::cuda::DeviceInfo deviceInfo;
        std::cout << "CUDA Device(s) Compatible: " << deviceInfo.isCompatible() << '\n';

        // Initialize the detector with model path, model input size, labels text path, thresholds
        // and inference target(GPU/CPU)
        Inference detector("../model/yolo11s.onnx", cv::Size(640, 640), "../model/labels.txt",
                            { .modelScoreThreshold = 0.45f, .modelNMSThreshold = 0.50f },
                            InferenceTarget::GPU);

        // Open camera device
        // Your camera device id may be different so use 'v4l2-ctl --list-devices' on terminal
        // and update it to 1 or what ever available for you.
        cv::VideoCapture cap(0);
        if (cap.isOpened()) {
            // Print camera resolution and FPS
            std::cout << "Camera Width: " << cap.get(cv::CAP_PROP_FRAME_WIDTH) << " - "
                    << "Camera Height: " << cap.get(cv::CAP_PROP_FRAME_HEIGHT)
                    << " FPS: " << cap.get(cv::CAP_PROP_FPS) << '\n';

            cv::Mat frame;
            while (true) {
                // Capture frame
                if (cap.read(frame)) {
                    // Run inference on current frame
                    const auto detections = detector.runInference(frame);

                    if (!detections.empty()) {
                        // Display inference time
                        cv::putText(frame, "Inference Time: " + std::to_string(detector.getLastInferenceTime()) + " ms",
                                    cv::Point(5, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,255), 1);

                        // Draw detection results on frame
                        for (const auto& detection : detections) {
                            cv::rectangle(frame, detection.box, detection.color, 2);
                            cv::putText(frame, detection.className + " " + std::to_string(int(detection.confidence*100)) + "%",
                            cv::Point(detection.box.x, detection.box.y-5),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, detection.color, 1);
                        }
                    }
                    // Show processed frame
                    cv::imshow("YOLOv11/12 Real Time Detection", frame);

                    // Press ESC to exit
                    if (cv::waitKey(1) == 27) {
                        break;
                    }
                }
            }
        }
        cap.release();
        cv::destroyAllWindows();
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << '\n';
        return -1;
    }
    return 0;
}