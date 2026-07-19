#pragma once

#include <filesystem>
#include <memory>

#include <opencv2/core.hpp>

#include "types.hpp"

namespace facetrack {

struct TrackerConfig {
    std::filesystem::path faceModel;
    std::filesystem::path landmarkModel;
    int calibrationFrames = 45;
    bool drawLandmarks = true;
    bool diagnostics = false;
    bool enableTongue = true;
};

class NeuralFaceTracker {
public:
    explicit NeuralFaceTracker(TrackerConfig config);
    ~NeuralFaceTracker();

    NeuralFaceTracker(const NeuralFaceTracker&) = delete;
    NeuralFaceTracker& operator=(const NeuralFaceTracker&) = delete;

    TrackingResult process(cv::Mat& frame, double fps);
    void recalibrate();
    void checkModels();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace facetrack
