#ifndef INFERENCE_HPP_
#define INFERENCE_HPP_

#include <fstream>
#include <random>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <algorithm>

/**
 * @struct InferenceThreshold
 * @brief Threshold values used in object detection
 *
 * Controls the sensitivity and accuracy of object detection.
 * Lower thresholds detect more objects but may include false positives.
 * Higher thresholds are more selective but may miss some objects.
 */
struct InferenceThreshold
{
    float modelScoreThreshold{};
    float modelNMSThreshold{};
};

/**
 * @struct Detection
 * @brief Holds complete information about a single detected object
 *
 * Stores all information needed for visualization and post processing
 * of detected objects in the frame.
 */
struct Detection
{
    int classId{};              ///< COCO class ID (0-79)
    std::string className{};    ///< Readable class name
    float confidence{};         ///< Detection confidence score
    cv::Scalar color{};         ///< BGR color for visualization
    cv::Rect box{};             ///< Bounding box in pixel coordinates
};

/**
 * @struct PaddingInfo
 * @brief Preprocessing transformation parameters
 *
 * Stores the letterbox padding information needed to map
 * detection coordinates back to the original frame.
 */
struct PaddingInfo
{
    cv::Mat paddedFrame{};  ///< Square frame with gray padding
    float scale{};          ///< Resize scale factor
    int top{};              ///< Top padding in pixels
    int left{};             ///< Left padding in pixels
};

/**
 * @enum InferenceTarget
 * @brief Specifies the compute target for inference
 *
 * GPU uses CUDA backend (requires OpenCV built with CUDA support)
 * CPU uses OpenCV's default CPU backend
 */
enum class InferenceTarget {
    GPU,
    CPU
};

/**
 * @class Inference
 * @brief YOLOv11/12 ONNX model inference
 */
class Inference
{
public:
    /**
     * @brief Constructor initializes the model with specified parameters
     * @param onnxModelPath Path to the ONNX model file
     * @param modelInputSize Model input size (640x640)
     * @param labelsPath Path to text file containing labels
     * @param threshold Detection thresholds for filtering
     * @param target Inference target (CUDA(GPU) / CPU)
     * @throws std::runtime_error if model or labels can not be loaded
     */
    explicit Inference(const std::string& onnxModelPath, const cv::Size& modelInputSize,
                        const std::string& labelsPath, const InferenceThreshold& threshold,
                        const InferenceTarget target);


    Inference(const Inference&) = delete;
    Inference& operator=(const Inference&) = delete;

    /**
     * @brief Performs object detection on a single frame
     * @param originalFrame Input frame
     * @return Vector of Detection objects containing filtered results
     */
    std::vector<Detection> runInference(const cv::Mat& originalFrame);

    /**
     * @brief Get the duration of the last inference operation
     * @return Inference time in milliseconds (excluding pre/postprocessing)
     */
    float getLastInferenceTime() const noexcept { return lastInferenceTimeMs_; }

private:
    /**
     * @brief Load COCO labels from text file
     * @throws std::runtime_error If file cannot be opened or has wrong number of labels
     * @note Expects exactly 80 labels for COCO dataset compatibility
     */
    void loadLabels();

    /**
     * @brief Initializes model with specified backend
     * @throws cv::Exception if model file is invalid
     */
    void loadYoloONNX();

    /**
     * @brief Applies letterbox padding for square input
     * @param sourceFrame Original frame of any aspect ratio
     * @return PaddingInfo containing padded frame and transformation parameters
     *
     * Letterbox padding ensures:
     *  - Aspect ratio is preserved (no distortion)
     *  - Frame is centered in square canvas
     *  - Padding uses gray (114) to match YOLO training
     */
    PaddingInfo letterboxPadding(const cv::Mat& sourceFrame) const;

    const std::string onnxModelPath_;
    const cv::Size modelInputSize_;
    const std::string labelsPath_;
    const float modelScoreThreshold_;
    const float modelNMSThreshold_;
    const InferenceTarget target_;

    std::vector<std::string> labelNames_{};
    cv::dnn::Net net_;

    float lastInferenceTimeMs_{};

    static constexpr int COCO_NUM_LABELS{80};
    static constexpr int YOLO_V11_12_OUTPUT_DIM{84};           ///< 4 bbox + 80 labels
    static constexpr int YOLO_V11_12_NUM_PREDICTIONS{8400};    ///< Total predictions
};

#endif