#include "Inference.hpp"

Inference::Inference(const std::string& onnxModelPath, const cv::Size& modelInputSize,
                        const std::string& labelsPath, const InferenceThreshold& threshold,
                        const InferenceTarget target)
    :
    onnxModelPath_{onnxModelPath}, modelInputSize_{modelInputSize}, labelsPath_{labelsPath},
    modelScoreThreshold_{threshold.modelScoreThreshold}, modelNMSThreshold_{threshold.modelNMSThreshold},
    target_{target}
{
    // Load model and set Inference Target GPU/CPU
    loadYoloONNX();
    // Load labels from file
    loadLabels();
}

void Inference::loadYoloONNX()
{
    // Read model using OpenCV DNN
    net_ = cv::dnn::readNetFromONNX(onnxModelPath_);

    if (target_ == InferenceTarget::GPU) {
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    } else if (target_ == InferenceTarget::CPU) {
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
}

void Inference::loadLabels()
{
    std::ifstream ifs(labelsPath_);
    if (!ifs.is_open()) {
        throw std::runtime_error("Can not open: " + labelsPath_);
    }

    // Read each label and store
    std::string line;
    while (std::getline(ifs, line)) {
        if (!line.empty()) {
            labelNames_.push_back(line);
        }
    }

    // Validate we have exactly 80 COCO labels
    if (labelNames_.size() != COCO_NUM_LABELS) {
        throw std::runtime_error("Expected 80 labels, got " +
                                std::to_string(labelNames_.size()));
    }
}

PaddingInfo Inference::letterboxPadding(const cv::Mat& sourceFrame) const
{
    /**
     * Letterbox Padding:
     *  - YOLO models are trained on square inputs (640x640)
     *  - Direct resize would distort objects and reduce accuracy
     *  - Letterbox maintains aspect ratio by adding gray padding
     *  - Used Gray (114), It's YOLO's training padding color
     */
    PaddingInfo info;
    const int sourceWidth = sourceFrame.cols;
    const int sourceHeight = sourceFrame.rows;
    // Ensures we get the larger dimension for square output
    const int targetSize = std::max(modelInputSize_.width, modelInputSize_.height);

    // Calculate scale to fit frame within square
    info.scale = static_cast<float>(targetSize) / std::max(sourceWidth, sourceHeight);
    // New size after scaling
    const int scaledWidth = static_cast<int>(sourceWidth * info.scale);
    const int scaledHeight = static_cast<int>(sourceHeight * info.scale);

    // Resize frame maintaining aspect ratio
    cv::Mat resized;
    cv::resize(sourceFrame, resized, cv::Size(scaledWidth, scaledHeight));

    // Calculate padding to center the resized frame
    info.top = (targetSize - scaledHeight) / 2;
    info.left = (targetSize - scaledWidth) / 2;

    // Apply padding with gray color (114,114,114)
    cv::copyMakeBorder(resized, info.paddedFrame,
                       info.top, targetSize - scaledHeight - info.top,
                       info.left, targetSize - scaledWidth - info.left,
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    return info;
}

std::vector<Detection> Inference::runInference(const cv::Mat& originalFrame)
{
    // Apply letterbox if the frame size is not equal to model input
    PaddingInfo padded = (originalFrame.rows != modelInputSize_.height || originalFrame.cols != modelInputSize_.width)
                         ? letterboxPadding(originalFrame)
                         : PaddingInfo{originalFrame, 1.0f, 0, 0};

    // Convert to blob
    cv::Mat blob;
    cv::dnn::blobFromImage(padded.paddedFrame, blob, 1.0/255.0, modelInputSize_, cv::Scalar(), true, false, CV_32F);
    net_.setInput(blob);

    // Forward pass through the net and measure inference time
    std::vector<cv::Mat> outputs;
    const auto inferenceStart = std::chrono::high_resolution_clock::now();
    net_.forward(outputs, net_.getUnconnectedOutLayersNames());
    const auto inferenceEnd = std::chrono::high_resolution_clock::now();
    lastInferenceTimeMs_ = std::chrono::duration<float, std::milli>(inferenceEnd - inferenceStart).count();

    if (!outputs.empty()) {
        /**
         * YOLOv11/12 Output Format:
         *
         * Raw output tensor shape: [1, 84, 8400]
         *  - Batch: 1
         *  - Features: 84 = 4 bbox coords (cx, cy, w, h) + 80 class scores
         *  - Predictions: 8400 = total candidate detections
         *
         * To use this output,
         * Transpose -> [8400, 84], so each row is one detection:
         *              [cx, cy, w, h, class0_score, class1_score, ..., class79_score]
         *
         */
        cv::Mat output;
        int numDetections, numFeatures;

        // Check if transpose is needed based on output dimensions
        if (outputs[0].size[2] == YOLO_V11_12_NUM_PREDICTIONS ||
            outputs[0].size[1] == YOLO_V11_12_OUTPUT_DIM) {
            // Reshape and transpose for row wise access/processing
            numFeatures = outputs[0].size[1];   // 84 features
            numDetections = outputs[0].size[2]; // 8400 detections
            output = outputs[0].reshape(0, numFeatures);
            cv::transpose(output, output);
            // Update dimensions after transpose
            numDetections = output.rows;
            numFeatures = output.cols;
        } else {
            // No need Transpose, already in correct format
            output = outputs[0];
            numDetections = outputs[0].size[1];
            numFeatures = outputs[0].size[2];
        }

        // Get pointer to raw float data for efficient sequential access.
        // cv::Mat::ptr<float>() is preferred over reinterpret_cast<float*>(output.data)
        // as it provides type safety and respects matrix layout
        const float* data = output.ptr<float>();

        // Calculate inverse transformation factors for mapping back to original frame
        // These factors undo the scaling and padding applied during preprocessing
        const float xFactor = 1.0f / padded.scale;
        const float yFactor = 1.0f / padded.scale;
        const int xOffset = padded.left;
        const int yOffset = padded.top;

        // Containers for all detections before NMS filtering
        std::vector<int> classIds;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;

        // Reserve space for efficiency to avoid reallocation
        classIds.reserve(numDetections);
        confidences.reserve(numDetections);
        boxes.reserve(numDetections);

        // Process each detection
        for (int i = 0; i < numDetections; ++i) {
            const float* rowData = data + i * numFeatures;

            // Extract bounding box
            const float centerX = rowData[0];
            const float centerY = rowData[1];
            const float width = rowData[2];
            const float height = rowData[3];

            // Class scores start at index 4 (after bounding box coordinates)
            const float* scores = rowData + 4;

            // Find class with highest confidence score
            const auto maxScoreIt = std::max_element(scores, scores + labelNames_.size());
            const auto bestClassId = std::distance(scores, maxScoreIt);  // Index of max element
            const float bestScore = *maxScoreIt;  // Value of max element

            if (bestScore > modelScoreThreshold_) {
                // Store detection info for NMS processing
                confidences.push_back(bestScore);
                classIds.push_back(static_cast<int>(bestClassId));

                // Convert YOLO format (center_x, center_y, width, height)
                // to OpenCV format (left, top, width, height).
                // Coordinates are adjusted to account for letterbox padding.
                const int left = static_cast<int>((centerX - 0.5f * width - xOffset) * xFactor);
                const int top = static_cast<int>((centerY - 0.5f * height - yOffset) * yFactor);
                const int boxWidth = static_cast<int>(width * xFactor);
                const int boxHeight = static_cast<int>(height * yFactor);

                /**
                 * Note:
                 * No clamping is applied here to keep boxes within frame boundaries.
                 * Since we aren't using these boxes for ROI/cropping but only for drawing,
                 * negative or out of bound values are safe.
                 * OpenCV's cv::rectangle automatically clips to the frame region,
                 * so there is no crash risk.
                 */

                boxes.emplace_back(left, top, boxWidth, boxHeight);
            }
        }

        // Apply NMS to remove overlapping detections
        // Keeps highest confidence detection when boxes overlap > threshold
        std::vector<int> nmsIndices;
        cv::dnn::NMSBoxes(boxes, confidences, modelScoreThreshold_, modelNMSThreshold_, nmsIndices);

        // Build final detection results
        std::vector<Detection> detections;
        detections.reserve(nmsIndices.size());

        // Random number generator for consistent colors per detection
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> colorDist(100, 255);

        for (const auto idx : nmsIndices) {
            Detection detection;
            detection.classId = classIds[idx];
            detection.confidence = confidences[idx];
            detection.box = boxes[idx];
            detection.className = labelNames_[detection.classId];
            detection.color = cv::Scalar(colorDist(gen), colorDist(gen), colorDist(gen));

            detections.push_back(std::move(detection));
        }
        return detections;
    }
    return std::vector<Detection>{};
}