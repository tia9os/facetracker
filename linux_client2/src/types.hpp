#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <string>

#include <opencv2/core.hpp>

namespace facetrack {

inline double clamp01(double value) {
    if (!std::isfinite(value)) {
        return 0.0;
    }
    return std::clamp(value, 0.0, 1.0);
}

inline std::int64_t epochMillis() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

enum class MouthState { Neutral, Happy, Sad, Surprised, Talking, Funny };
enum class EyeState { Open, Closed, Focused };
enum class BrowState { Neutral, Raised, Sad, Focused };
enum class Expression {
    Neutral, Happy, Funny, Sad, Surprised, Talking, Blinking, Winking, Focused, NoFace
};

struct FaceParts {
    MouthState mouth = MouthState::Neutral;
    EyeState leftEye = EyeState::Open;
    EyeState rightEye = EyeState::Open;
    BrowState leftBrow = BrowState::Neutral;
    BrowState rightBrow = BrowState::Neutral;
};

struct Scores {
    double smile = 0.0;
    double mouthOpen = 0.0;
    double mouthWide = 0.0;
    double sad = 0.0;
    double tongue = 0.0;
    double blink = 0.0;
    double wink = 0.0;
    double leftClosed = 0.0;
    double rightClosed = 0.0;
    double leftNarrow = 0.0;
    double rightNarrow = 0.0;
    double mouthActivity = 0.0;
    double browRaiseLeft = 0.0;
    double browRaiseRight = 0.0;
    double browSadLeft = 0.0;
    double browSadRight = 0.0;
    double browFurrow = 0.0;
};

struct HeadPose {
    bool valid = false;
    double pitch = 0.0;
    double yaw = 0.0;
    double roll = 0.0;
};

struct TrackingResult {
    std::int64_t timestamp = epochMillis();
    bool faceFound = false;
    int faceCount = 0;
    cv::Rect face;
    double confidence = 0.0;
    double detectionConfidence = 0.0;
    double landmarkConfidence = 0.0;
    double trackingQuality = 0.0;
    double fps = 0.0;
    bool calibrated = false;
    double calibrationProgress = 0.0;
    std::string status;
    HeadPose pose;
    Scores scores;
    FaceParts parts;
    Expression expression = Expression::NoFace;
};

const char* wire(MouthState value);
const char* wire(EyeState value);
const char* wire(BrowState value);
const char* wire(Expression value);
Expression expressionFor(const FaceParts& parts);

}  // namespace facetrack
