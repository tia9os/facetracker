#include "tracker.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/face.hpp>

namespace facetrack {

namespace {

constexpr int kLandmarkCount = 68;
constexpr int kFanInputSize = 256;
constexpr int kDetectorMaxDimension = 640;
constexpr int kDetectorStride = 32;

int alignUp(int value, int alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

double distance(const std::vector<cv::Point2f>& points, int first, int second) {
    cv::Point2f delta = points.at(static_cast<std::size_t>(first))
            - points.at(static_cast<std::size_t>(second));
    return std::hypot(static_cast<double>(delta.x), static_cast<double>(delta.y));
}

double averageY(const std::vector<cv::Point2f>& points, int first, int last) {
    double sum = 0.0;
    for (int index = first; index <= last; ++index) {
        sum += points.at(static_cast<std::size_t>(index)).y;
    }
    return sum / static_cast<double>(last - first + 1);
}

double eyeAspectRatio(const std::vector<cv::Point2f>& points, int left, int upperLeft,
                      int upperRight, int right, int lowerRight, int lowerLeft) {
    double horizontal = distance(points, left, right);
    if (horizontal < 1.0) {
        return 0.0;
    }
    double vertical = distance(points, upperLeft, lowerLeft)
            + distance(points, upperRight, lowerRight);
    return vertical / (2.0 * horizontal);
}

cv::Rect boundedRect(int x, int y, int width, int height, int frameWidth, int frameHeight) {
    int left = std::clamp(x, 0, std::max(0, frameWidth - 1));
    int top = std::clamp(y, 0, std::max(0, frameHeight - 1));
    int right = std::clamp(x + std::max(1, width), left + 1, frameWidth);
    int bottom = std::clamp(y + std::max(1, height), top + 1, frameHeight);
    return {left, top, right - left, bottom - top};
}

cv::Rect expandedRect(const cv::Rect& rectangle, double horizontal, double vertical,
                      int frameWidth, int frameHeight) {
    int padX = static_cast<int>(std::lround(rectangle.width * horizontal));
    int padY = static_cast<int>(std::lround(rectangle.height * vertical));
    return boundedRect(rectangle.x - padX, rectangle.y - padY,
                       rectangle.width + padX * 2, rectangle.height + padY * 2,
                       frameWidth, frameHeight);
}

double intersectionOverUnion(const cv::Rect& first, const cv::Rect& second) {
    cv::Rect overlap = first & second;
    double overlapArea = static_cast<double>(std::max(0, overlap.area()));
    double unionArea = static_cast<double>(first.area())
            + static_cast<double>(second.area()) - overlapArea;
    return unionArea > 0.0 ? clamp01(overlapArea / unionArea) : 0.0;
}

cv::Rect blendRect(const cv::Rect& previous, const cv::Rect& current, double alpha,
                   int frameWidth, int frameHeight) {
    double keep = 1.0 - clamp01(alpha);
    return boundedRect(
            static_cast<int>(std::lround(previous.x * keep + current.x * alpha)),
            static_cast<int>(std::lround(previous.y * keep + current.y * alpha)),
            static_cast<int>(std::lround(previous.width * keep + current.width * alpha)),
            static_cast<int>(std::lround(previous.height * keep + current.height * alpha)),
            frameWidth, frameHeight
    );
}

std::string fixed(double value, int digits = 2) {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(digits) << value;
    return stream.str();
}

double median(std::vector<double> values) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    std::size_t middle = values.size() / 2;
    if (values.size() % 2 == 0) {
        return (values[middle - 1] + values[middle]) * 0.5;
    }
    return values[middle];
}

double asymmetricSmooth(double previous, double current, double rise, double fall) {
    double alpha = current > previous ? rise : fall;
    return previous + (current - previous) * clamp01(alpha);
}

template <typename T>
class StableChoice {
public:
    explicit StableChoice(T initial) : stable_(initial), pending_(initial) {
    }

    T update(T requested, int requiredFrames) {
        if (requested == stable_) {
            pending_ = stable_;
            pendingFrames_ = 0;
            return stable_;
        }
        if (requested == pending_) {
            ++pendingFrames_;
        } else {
            pending_ = requested;
            pendingFrames_ = 1;
        }
        if (pendingFrames_ >= std::max(1, requiredFrames)) {
            stable_ = pending_;
            pendingFrames_ = 0;
        }
        return stable_;
    }

    void reset(T value) {
        stable_ = value;
        pending_ = value;
        pendingFrames_ = 0;
    }

private:
    T stable_;
    T pending_;
    int pendingFrames_ = 0;
};

struct FaceDetection {
    bool found = false;
    bool predicted = false;
    cv::Rect rectangle;
    int count = 0;
    double confidence = 0.0;
};

struct Geometry {
    double leftEar = 0.0;
    double rightEar = 0.0;
    double mouthOpen = 0.0;
    double mouthWidth = 0.0;
    double mouthCurve = 0.0;
    double leftBrowGap = 0.0;
    double rightBrowGap = 0.0;
    double browPinch = 0.0;
    double leftBrowSlope = 0.0;
    double rightBrowSlope = 0.0;
};

struct Baseline {
    bool valid = false;
    Geometry neutral;
};

class Calibrator {
public:
    explicit Calibrator(int requiredFrames) : requiredFrames_(std::max(1, requiredFrames)) {
    }

    void reset() {
        samples_.clear();
        baseline_ = {};
    }

    bool add(const Geometry& geometry, const HeadPose& pose, double quality) {
        if (baseline_.valid || quality < 0.48) {
            return false;
        }
        if (pose.valid && (std::abs(pose.yaw) > 30.0 || std::abs(pose.pitch) > 30.0
                || std::abs(pose.roll) > 24.0)) {
            return false;
        }
        double largerEar = std::max(geometry.leftEar, geometry.rightEar);
        double smallerEar = std::min(geometry.leftEar, geometry.rightEar);
        if (smallerEar < 0.055 || smallerEar / std::max(0.055, largerEar) < 0.58
                || geometry.mouthOpen > 0.24) {
            return false;
        }
        samples_.push_back(geometry);
        if (static_cast<int>(samples_.size()) >= requiredFrames_) {
            finish();
        }
        return true;
    }

    double progress() const {
        if (baseline_.valid) {
            return 1.0;
        }
        return clamp01(static_cast<double>(samples_.size()) / static_cast<double>(requiredFrames_));
    }

    const Baseline& baseline() const {
        return baseline_;
    }

private:
    void finish() {
        auto collect = [this](auto member) {
            std::vector<double> values;
            values.reserve(samples_.size());
            for (const Geometry& sample : samples_) {
                values.push_back(sample.*member);
            }
            return median(std::move(values));
        };
        baseline_.neutral.leftEar = collect(&Geometry::leftEar);
        baseline_.neutral.rightEar = collect(&Geometry::rightEar);
        baseline_.neutral.mouthOpen = collect(&Geometry::mouthOpen);
        baseline_.neutral.mouthWidth = collect(&Geometry::mouthWidth);
        baseline_.neutral.mouthCurve = collect(&Geometry::mouthCurve);
        baseline_.neutral.leftBrowGap = collect(&Geometry::leftBrowGap);
        baseline_.neutral.rightBrowGap = collect(&Geometry::rightBrowGap);
        baseline_.neutral.browPinch = collect(&Geometry::browPinch);
        baseline_.neutral.leftBrowSlope = collect(&Geometry::leftBrowSlope);
        baseline_.neutral.rightBrowSlope = collect(&Geometry::rightBrowSlope);
        baseline_.valid = true;
    }

    int requiredFrames_;
    std::vector<Geometry> samples_;
    Baseline baseline_;
};

Geometry measureGeometry(const std::vector<cv::Point2f>& points, const cv::Rect& face) {
    double faceWidth = std::max(1.0, static_cast<double>(face.width));
    double faceHeight = std::max(1.0, static_cast<double>(face.height));
    double mouthWidthPixels = std::max(1.0, distance(points, 48, 54));
    double innerMouthHeight = (distance(points, 61, 67)
            + distance(points, 62, 66)
            + distance(points, 63, 65)) / 3.0;
    double cornerY = (points[48].y + points[54].y) * 0.5;
    double lipCenterY = (points[51].y + points[57].y + points[62].y + points[66].y) * 0.25;

    Geometry geometry;
    geometry.leftEar = eyeAspectRatio(points, 36, 37, 38, 39, 40, 41);
    geometry.rightEar = eyeAspectRatio(points, 42, 43, 44, 45, 46, 47);
    geometry.mouthOpen = innerMouthHeight / mouthWidthPixels;
    geometry.mouthWidth = mouthWidthPixels / faceWidth;
    geometry.mouthCurve = (lipCenterY - cornerY) / faceHeight;
    geometry.leftBrowGap = (averageY(points, 36, 41) - averageY(points, 17, 21)) / faceHeight;
    geometry.rightBrowGap = (averageY(points, 42, 47) - averageY(points, 22, 26)) / faceHeight;
    geometry.browPinch = distance(points, 21, 22) / faceWidth;
    geometry.leftBrowSlope = (points[21].y - points[17].y) / faceHeight;
    geometry.rightBrowSlope = (points[22].y - points[26].y) / faceHeight;
    return geometry;
}

bool plausibleLandmarks(const std::vector<cv::Point2f>& points, const cv::Rect& face,
                        int frameWidth, int frameHeight) {
    if (points.size() != kLandmarkCount || face.area() <= 0) {
        return false;
    }
    cv::Rect allowed = expandedRect(face, 0.32, 0.38, frameWidth, frameHeight);
    float allowedLeft = static_cast<float>(allowed.x);
    float allowedTop = static_cast<float>(allowed.y);
    float allowedRight = static_cast<float>(allowed.x + allowed.width);
    float allowedBottom = static_cast<float>(allowed.y + allowed.height);
    int outside = 0;
    for (const cv::Point2f& point : points) {
        if (!std::isfinite(point.x) || !std::isfinite(point.y)) {
            return false;
        }
        if (point.x < allowedLeft || point.x > allowedRight
                || point.y < allowedTop || point.y > allowedBottom) {
            ++outside;
        }
    }
    if (outside > 3) {
        return false;
    }
    Geometry geometry = measureGeometry(points, face);
    double jawWidth = distance(points, 0, 16) / std::max(1.0, static_cast<double>(face.width));
    double eyeLine = (averageY(points, 36, 41) + averageY(points, 42, 47)) * 0.5;
    double mouthLine = averageY(points, 48, 59);
    return geometry.leftEar >= 0.035 && geometry.leftEar <= 0.60
            && geometry.rightEar >= 0.035 && geometry.rightEar <= 0.60
            && geometry.mouthWidth >= 0.16 && geometry.mouthWidth <= 0.66
            && geometry.mouthOpen >= 0.0 && geometry.mouthOpen <= 0.85
            && jawWidth >= 0.44 && jawWidth <= 1.40
            && points[36].x < points[45].x
            && mouthLine > eyeLine + face.height * 0.12;
}

HeadPose estimateHeadPose(const std::vector<cv::Point2f>& points, const cv::Size& frameSize) {
    static const std::vector<cv::Point3d> model{
            {0.0, 0.0, 0.0},
            {0.0, -330.0, -65.0},
            {-225.0, 170.0, -135.0},
            {225.0, 170.0, -135.0},
            {-150.0, -150.0, -125.0},
            {150.0, -150.0, -125.0},
    };
    std::vector<cv::Point2d> image{
            points[30], points[8], points[36], points[45], points[48], points[54]
    };
    double focalLength = static_cast<double>(std::max(frameSize.width, frameSize.height));
    cv::Matx33d camera(
            focalLength, 0.0, frameSize.width * 0.5,
            0.0, focalLength, frameSize.height * 0.5,
            0.0, 0.0, 1.0
    );
    cv::Vec<double, 5> distortion = cv::Vec<double, 5>::all(0.0);
    cv::Vec3d rotationVector;
    cv::Vec3d translationVector;
    bool solved = cv::solvePnP(model, image, camera, distortion, rotationVector, translationVector,
                               false, cv::SOLVEPNP_ITERATIVE);
    if (!solved) {
        return {};
    }
    cv::Matx33d rotation;
    cv::Rodrigues(rotationVector, rotation);
    cv::Matx33d matrixR;
    cv::Matx33d matrixQ;
    cv::Vec3d euler = cv::RQDecomp3x3(rotation, matrixR, matrixQ);
    HeadPose pose;
    pose.valid = std::isfinite(euler[0]) && std::isfinite(euler[1]) && std::isfinite(euler[2]);
    auto wrapDegrees = [](double angle) {
        while (angle > 180.0) angle -= 360.0;
        while (angle < -180.0) angle += 360.0;
        return angle;
    };
    pose.pitch = wrapDegrees(euler[0]);
    pose.yaw = wrapDegrees(euler[1]);
    pose.roll = wrapDegrees(euler[2]);
    // The conventional six-point face model has its vertical axis opposite
    // OpenCV's camera axis, which can express a normal pose as pitch/roll
    // near 180 degrees. Convert that equivalent rotation to human-readable
    // angles before quality gating.
    if (pose.pitch > 90.0) {
        pose.pitch -= 180.0;
        pose.roll = wrapDegrees(pose.roll + 180.0);
    } else if (pose.pitch < -90.0) {
        pose.pitch += 180.0;
        pose.roll = wrapDegrees(pose.roll + 180.0);
    }
    return pose;
}

}  // namespace

class NeuralFaceTracker::Impl {
public:
    explicit Impl(TrackerConfig config)
            : config_(std::move(config)), calibrator_(config_.calibrationFrames) {
        if (!std::filesystem::is_regular_file(config_.faceModel)) {
            throw std::runtime_error("YuNet face model not found: " + config_.faceModel.string());
        }
        if (!std::filesystem::is_regular_file(config_.landmarkModel)) {
            throw std::runtime_error("FAN landmark model not found: " + config_.landmarkModel.string());
        }

        detector_ = cv::FaceDetectorYN::create(
                config_.faceModel.string(), "", cv::Size(640, 480),
                0.72F, 0.30F, 5000,
                cv::dnn::DNN_BACKEND_OPENCV, cv::dnn::DNN_TARGET_CPU
        );
        landmarkNet_ = cv::dnn::readNetFromONNX(config_.landmarkModel.string());
        landmarkNet_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        landmarkNet_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        checkModels();
    }

    void checkModels() {
        cv::Mat detectorInput(320, 320, CV_8UC3, cv::Scalar::all(0));
        detector_->setInputSize(detectorInput.size());
        cv::Mat detectorOutput;
        detector_->detect(detectorInput, detectorOutput);

        int shape[] = {1, 3, kFanInputSize, kFanInputSize};
        cv::Mat input(4, shape, CV_32F, cv::Scalar(0.0F));
        landmarkNet_.setInput(input);
        cv::Mat output = landmarkNet_.forward("heatmaps");
        constexpr std::size_t expected = static_cast<std::size_t>(kLandmarkCount * 64 * 64);
        if (output.total() != expected || output.dims != 4) {
            throw std::runtime_error("FAN heatmap output must be [1,68,64,64], received "
                                     + std::to_string(output.total()) + " values");
        }
    }

    void recalibrate() {
        calibrator_.reset();
        resetSignals();
    }

    TrackingResult process(cv::Mat& frame, double fps) {
        TrackingResult result;
        result.fps = fps;

        FaceDetection detection = detectFace(frame);
        result.faceCount = detection.count;
        result.face = detection.rectangle;
        result.detectionConfidence = detection.confidence;

        if (!detection.found) {
            ++missingFaceFrames_;
            if (missingFaceFrames_ >= 3) {
                resetStates();
            }
            result.status = "No face";
            annotate(frame, result, {});
            return result;
        }
        missingFaceFrames_ = 0;
        result.faceFound = true;

        std::vector<cv::Point2f> landmarks;
        double landmarkConfidence = 0.0;
        if (!inferLandmarks(frame, detection.rectangle, landmarks, landmarkConfidence)) {
            ++badLandmarkFrames_;
            result.landmarkConfidence = landmarkConfidence;
            result.confidence = detection.confidence * 0.25;
            result.status = "Face found - neural landmarks uncertain";
            result.parts = badLandmarkFrames_ <= 2 ? previousParts_ : FaceParts{};
            result.expression = expressionFor(result.parts);
            annotate(frame, result, {});
            return result;
        }
        badLandmarkFrames_ = 0;
        smoothLandmarks(landmarks, detection.rectangle);

        result.landmarkConfidence = landmarkConfidence;
        result.pose = estimateHeadPose(landmarks, frame.size());
        double poseQuality = poseReliability(result.pose);
        result.trackingQuality = clamp01(
                detection.confidence * 0.32 + landmarkConfidence * 0.54 + poseQuality * 0.14
        );

        Geometry geometry = measureGeometry(landmarks, detection.rectangle);
        if (!calibrator_.baseline().valid) {
            bool accepted = calibrator_.add(geometry, result.pose, result.trackingQuality);
            if (config_.diagnostics && !accepted && !calibrationDiagnosticReported_) {
                calibrationDiagnosticReported_ = true;
                std::cerr << "Calibration diagnostic: q " << result.trackingQuality
                          << ", det " << detection.confidence
                          << ", lm " << landmarkConfidence
                          << ", pose " << result.pose.pitch << "/"
                          << result.pose.yaw << "/" << result.pose.roll
                          << ", EAR " << geometry.leftEar << "/" << geometry.rightEar
                          << ", mouth open " << geometry.mouthOpen << '\n';
            } else if (accepted) {
                calibrationDiagnosticReported_ = false;
            }
            result.calibrated = calibrator_.baseline().valid;
            result.calibrationProgress = calibrator_.progress();
            int percent = static_cast<int>(std::lround(result.calibrationProgress * 100.0));
            result.status = accepted
                    ? "Calibrating neutral face: " + std::to_string(percent) + "%"
                    : "Calibration: face camera, eyes open, mouth relaxed";
            result.confidence = result.trackingQuality * 0.45;
            result.parts = {};
            result.expression = Expression::Neutral;
            previousParts_ = result.parts;
            annotate(frame, result, landmarks);
            return result;
        }

        result.calibrated = true;
        result.calibrationProgress = 1.0;
        Scores rawScores = scoreGeometry(geometry, calibrator_.baseline().neutral);
        if (config_.enableTongue) {
            rawScores.tongue = detectTongue(frame, landmarks, rawScores.mouthOpen);
        } else {
            lastTongueBox_ = {};
        }
        dampenForPose(rawScores, result.pose);
        result.scores = smoothScores(rawScores);
        result.parts = classify(result.scores);
        result.expression = expressionFor(result.parts);
        result.confidence = result.trackingQuality;
        result.status = result.trackingQuality >= 0.58 ? "Tracking" : "Tracking quality low";
        previousGeometry_ = geometry;
        geometryInitialized_ = true;
        annotate(frame, result, landmarks);
        return result;
    }

private:
    FaceDetection detectFace(const cv::Mat& frame) {
        double detectorScale = std::min(
                1.0,
                static_cast<double>(kDetectorMaxDimension)
                        / static_cast<double>(std::max(frame.cols, frame.rows))
        );
        int resizedWidth = std::max(
                1, static_cast<int>(std::lround(frame.cols * detectorScale))
        );
        int resizedHeight = std::max(
                1, static_cast<int>(std::lround(frame.rows * detectorScale))
        );
        int detectorWidth = alignUp(resizedWidth, kDetectorStride);
        int detectorHeight = alignUp(resizedHeight, kDetectorStride);
        int padLeft = (detectorWidth - resizedWidth) / 2;
        int padRight = detectorWidth - resizedWidth - padLeft;
        int padTop = (detectorHeight - resizedHeight) / 2;
        int padBottom = detectorHeight - resizedHeight - padTop;

        cv::Mat resized;
        cv::resize(frame, resized, cv::Size(resizedWidth, resizedHeight),
                   0.0, 0.0, cv::INTER_LINEAR);
        cv::Mat detectorFrame;
        cv::copyMakeBorder(resized, detectorFrame, padTop, padBottom, padLeft, padRight,
                           cv::BORDER_CONSTANT, cv::Scalar::all(0));

        detector_->setInputSize(detectorFrame.size());
        cv::Mat detections;
        detector_->detect(detectorFrame, detections);

        FaceDetection best;
        double bestRank = -1.0;
        int validCount = 0;
        for (int row = 0; row < detections.rows; ++row) {
            if (detections.cols < 15) {
                break;
            }
            double confidence = detections.at<float>(row, 14);
            cv::Rect rectangle = boundedRect(
                    static_cast<int>(std::lround(
                            (static_cast<double>(detections.at<float>(row, 0)) - padLeft)
                                    / detectorScale
                    )),
                    static_cast<int>(std::lround(
                            (static_cast<double>(detections.at<float>(row, 1)) - padTop)
                                    / detectorScale
                    )),
                    static_cast<int>(std::lround(
                            static_cast<double>(detections.at<float>(row, 2)) / detectorScale
                    )),
                    static_cast<int>(std::lround(
                            static_cast<double>(detections.at<float>(row, 3)) / detectorScale
                    )),
                    frame.cols, frame.rows
            );
            double areaRatio = static_cast<double>(rectangle.area())
                    / std::max(1.0, static_cast<double>(frame.cols * frame.rows));
            double aspect = static_cast<double>(rectangle.width)
                    / std::max(1.0, static_cast<double>(rectangle.height));
            if (areaRatio < 0.004 || areaRatio > 0.82 || aspect < 0.48 || aspect > 1.35) {
                continue;
            }
            ++validCount;
            double continuity = lastFaceValid_ ? intersectionOverUnion(lastFace_, rectangle) : 0.35;
            double sizeScore = clamp01(std::sqrt(areaRatio) / 0.42);
            double rank = confidence * 0.68 + continuity * 0.24 + sizeScore * 0.08;
            if (rank > bestRank) {
                bestRank = rank;
                best.found = true;
                best.rectangle = rectangle;
                best.confidence = confidence;
            }
        }
        best.count = validCount;

        if (best.found) {
            if (lastFaceValid_ && intersectionOverUnion(lastFace_, best.rectangle) > 0.08) {
                best.rectangle = blendRect(lastFace_, best.rectangle, 0.68, frame.cols, frame.rows);
            } else {
                lastLandmarks_.clear();
            }
            lastFace_ = best.rectangle;
            lastFaceValid_ = true;
            missedDetections_ = 0;
            return best;
        }

        ++missedDetections_;
        if (lastFaceValid_ && missedDetections_ <= 3) {
            best.found = true;
            best.predicted = true;
            best.rectangle = lastFace_;
            best.count = 1;
            best.confidence = std::max(0.22, 0.58 - missedDetections_ * 0.10);
            return best;
        }
        lastFaceValid_ = false;
        lastLandmarks_.clear();
        return {};
    }

    bool inferLandmarks(const cv::Mat& frame, const cv::Rect& face,
                        std::vector<cv::Point2f>& landmarks, double& confidence) {
        cv::Rect crop = expandedRect(face, 0.12, 0.14, frame.cols, frame.rows);
        cv::Mat blob = cv::dnn::blobFromImage(
                frame(crop), 1.0 / 255.0, cv::Size(kFanInputSize, kFanInputSize),
                cv::Scalar(), true, false, CV_32F
        );
        landmarkNet_.setInput(blob);
        cv::Mat heatmaps = landmarkNet_.forward("heatmaps");
        if (heatmaps.empty() || heatmaps.dims != 4 || heatmaps.size[0] != 1
                || heatmaps.size[1] != kLandmarkCount
                || heatmaps.size[2] <= 1 || heatmaps.size[3] <= 1
                || heatmaps.type() != CV_32F) {
            return false;
        }
        int heatmapHeight = heatmaps.size[2];
        int heatmapWidth = heatmaps.size[3];
        double scoreSum = 0.0;
        landmarks.clear();
        landmarks.reserve(kLandmarkCount);
        for (int index = 0; index < kLandmarkCount; ++index) {
            const float* data = heatmaps.ptr<float>(0, index);
            cv::Mat heatmap(heatmapHeight, heatmapWidth, CV_32F, const_cast<float*>(data));
            double maximum = 0.0;
            cv::Point peak;
            cv::minMaxLoc(heatmap, nullptr, &maximum, nullptr, &peak);
            if (!std::isfinite(maximum)) {
                return false;
            }
            double refinedX = peak.x;
            double refinedY = peak.y;
            if (peak.x > 0 && peak.x + 1 < heatmapWidth) {
                float left = heatmap.at<float>(peak.y, peak.x - 1);
                float right = heatmap.at<float>(peak.y, peak.x + 1);
                refinedX += right > left ? 0.25 : right < left ? -0.25 : 0.0;
            }
            if (peak.y > 0 && peak.y + 1 < heatmapHeight) {
                float above = heatmap.at<float>(peak.y - 1, peak.x);
                float below = heatmap.at<float>(peak.y + 1, peak.x);
                refinedY += below > above ? 0.25 : below < above ? -0.25 : 0.0;
            }
            double normalizedX = (refinedX + 0.5) / static_cast<double>(heatmapWidth);
            double normalizedY = (refinedY + 0.5) / static_cast<double>(heatmapHeight);
            landmarks.emplace_back(
                    static_cast<float>(crop.x + normalizedX * crop.width),
                    static_cast<float>(crop.y + normalizedY * crop.height)
            );
            scoreSum += clamp01(maximum);
        }
        confidence = scoreSum / static_cast<double>(kLandmarkCount);
        if (confidence < 0.18) {
            reportFanDiagnostics("low heatmap confidence", 0.0,
                                 static_cast<double>(heatmapWidth), confidence, {});
            return false;
        }
        bool plausible = plausibleLandmarks(landmarks, face, frame.cols, frame.rows);
        if (!plausible) {
            reportFanDiagnostics("implausible heatmap geometry", 0.0,
                                 static_cast<double>(heatmapWidth),
                                 confidence, landmarks);
        } else {
            diagnosticReported_ = false;
        }
        return plausible;
    }

    void reportFanDiagnostics(const char* reason, double coordinateMin, double coordinateMax,
                              double confidence, const std::vector<cv::Point2f>& landmarks) {
        if (!config_.diagnostics || diagnosticReported_) {
            return;
        }
        diagnosticReported_ = true;
        std::cerr << "FAN diagnostic: " << reason
                  << ", coordinate range " << coordinateMin << ".." << coordinateMax
                  << ", mean score " << confidence;
        if (landmarks.size() == kLandmarkCount) {
            Geometry geometry = measureGeometry(landmarks, lastFace_);
            std::cerr << ", EAR " << geometry.leftEar << "/" << geometry.rightEar
                      << ", mouth width " << geometry.mouthWidth
                      << ", mouth open " << geometry.mouthOpen
                      << ", jaw ratio "
                      << distance(landmarks, 0, 16)
                         / std::max(1.0, static_cast<double>(lastFace_.width));
        }
        std::cerr << '\n';
    }

    void smoothLandmarks(std::vector<cv::Point2f>& landmarks, const cv::Rect& face) {
        if (lastLandmarks_.size() == kLandmarkCount && lastLandmarkFace_.area() > 0
                && intersectionOverUnion(lastLandmarkFace_, face) > 0.08) {
            double oldWidth = std::max(1.0, static_cast<double>(lastLandmarkFace_.width));
            double oldHeight = std::max(1.0, static_cast<double>(lastLandmarkFace_.height));
            for (int index = 0; index < kLandmarkCount; ++index) {
                std::size_t position = static_cast<std::size_t>(index);
                double normalizedX = (static_cast<double>(lastLandmarks_[position].x)
                                      - static_cast<double>(lastLandmarkFace_.x)) / oldWidth;
                double normalizedY = (static_cast<double>(lastLandmarks_[position].y)
                                      - static_cast<double>(lastLandmarkFace_.y)) / oldHeight;
                cv::Point2f projected(
                        static_cast<float>(face.x + normalizedX * face.width),
                        static_cast<float>(face.y + normalizedY * face.height)
                );
                bool expressive = index >= 17;
                float alpha = expressive ? 0.76F : 0.62F;
                landmarks[position] = projected * (1.0F - alpha) + landmarks[position] * alpha;
            }
        }
        lastLandmarks_ = landmarks;
        lastLandmarkFace_ = face;
    }

    static double poseReliability(const HeadPose& pose) {
        if (!pose.valid) {
            return 0.45;
        }
        double yaw = clamp01(std::abs(pose.yaw) / 42.0);
        double pitch = clamp01(std::abs(pose.pitch) / 38.0);
        double roll = clamp01(std::abs(pose.roll) / 35.0);
        return clamp01(1.0 - yaw * 0.52 - pitch * 0.30 - roll * 0.18);
    }

    Scores scoreGeometry(const Geometry& geometry, const Geometry& neutral) {
        Scores scores;
        double leftRatio = geometry.leftEar / std::max(0.12, neutral.leftEar);
        double rightRatio = geometry.rightEar / std::max(0.12, neutral.rightEar);
        scores.leftClosed = clamp01((0.74 - leftRatio) / 0.28);
        scores.rightClosed = clamp01((0.74 - rightRatio) / 0.28);
        scores.leftNarrow = clamp01((0.88 - leftRatio) / 0.24) * (1.0 - scores.leftClosed);
        scores.rightNarrow = clamp01((0.88 - rightRatio) / 0.24) * (1.0 - scores.rightClosed);
        scores.blink = std::min(scores.leftClosed, scores.rightClosed);
        double leftOpen = 1.0 - scores.leftClosed;
        double rightOpen = 1.0 - scores.rightClosed;
        scores.wink = std::max(scores.leftClosed * rightOpen, scores.rightClosed * leftOpen);

        scores.mouthOpen = clamp01((geometry.mouthOpen - neutral.mouthOpen - 0.018) / 0.28);
        double widthRatio = geometry.mouthWidth / std::max(0.12, neutral.mouthWidth);
        scores.mouthWide = clamp01((widthRatio - 1.015) / 0.16);
        double smileCurve = clamp01((geometry.mouthCurve - neutral.mouthCurve - 0.003) / 0.032);
        scores.smile = clamp01(smileCurve * 0.62 + scores.mouthWide * 0.38
                               - scores.mouthOpen * 0.16);
        double sadCurve = clamp01((neutral.mouthCurve - geometry.mouthCurve - 0.003) / 0.030);
        double narrowMouth = clamp01((0.98 - widthRatio) / 0.12);
        scores.sad = clamp01(sadCurve * 0.78 + narrowMouth * 0.22
                             - scores.mouthOpen * 0.30);

        double rawActivity = 0.0;
        if (geometryInitialized_) {
            rawActivity = clamp01(
                    std::max(0.0, std::abs(geometry.mouthOpen - previousGeometry_.mouthOpen) - 0.008) * 5.4
                    + std::max(0.0, std::abs(geometry.mouthWidth - previousGeometry_.mouthWidth) - 0.004) * 5.0
            );
        }
        scores.mouthActivity = rawActivity;
        scores.browRaiseLeft = clamp01(
                (geometry.leftBrowGap - neutral.leftBrowGap - 0.004) / 0.034
        );
        scores.browRaiseRight = clamp01(
                (geometry.rightBrowGap - neutral.rightBrowGap - 0.004) / 0.034
        );
        scores.browFurrow = clamp01(
                (neutral.browPinch - geometry.browPinch - 0.003) / 0.042
        );
        scores.browSadLeft = clamp01(
                (neutral.leftBrowSlope - geometry.leftBrowSlope - 0.003) / 0.035
        );
        scores.browSadRight = clamp01(
                (neutral.rightBrowSlope - geometry.rightBrowSlope - 0.003) / 0.035
        );
        return scores;
    }

    double detectTongue(const cv::Mat& frame, const std::vector<cv::Point2f>& landmarks,
                        double mouthOpenScore) {
        lastTongueBox_ = {};
        if (frame.channels() < 3 || landmarks.size() != kLandmarkCount || mouthOpenScore < 0.16) {
            return 0.0;
        }

        const cv::Point2f& innerLeft = landmarks[60];
        const cv::Point2f& innerRight = landmarks[64];
        double mouthWidth = std::max(8.0, distance(landmarks, 48, 54));
        double lowerLipY = landmarks[55].y;
        for (int index = 56; index <= 59; ++index) {
            lowerLipY = std::max(lowerLipY, static_cast<double>(landmarks[static_cast<std::size_t>(index)].y));
        }
        float bottomY = static_cast<float>(std::min(
                static_cast<double>(frame.rows - 1),
                lowerLipY + mouthWidth * 0.72
        ));
        float bottomInset = static_cast<float>(mouthWidth * 0.13);
        std::vector<cv::Point2f> searchPolygon{
                innerLeft,
                landmarks[61],
                landmarks[62],
                landmarks[63],
                innerRight,
                cv::Point2f(innerRight.x - bottomInset, bottomY),
                cv::Point2f(innerLeft.x + bottomInset, bottomY),
        };
        cv::Rect rawBounds = cv::boundingRect(searchPolygon);
        cv::Rect bounds = boundedRect(rawBounds.x - 2, rawBounds.y - 2,
                                      rawBounds.width + 4, rawBounds.height + 4,
                                      frame.cols, frame.rows);
        if (bounds.width < 10 || bounds.height < 6) {
            return 0.0;
        }

        std::vector<cv::Point> localPolygon;
        localPolygon.reserve(searchPolygon.size());
        for (const cv::Point2f& point : searchPolygon) {
            localPolygon.emplace_back(
                    static_cast<int>(std::lround(point.x)) - bounds.x,
                    static_cast<int>(std::lround(point.y)) - bounds.y
            );
        }

        cv::Mat mouthMask = cv::Mat::zeros(bounds.size(), CV_8U);
        std::vector<std::vector<cv::Point>> polygons{localPolygon};
        cv::fillPoly(mouthMask, polygons, cv::Scalar(255), cv::LINE_AA);
        cv::Mat erosionKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
        cv::erode(mouthMask, mouthMask, erosionKernel);
        double mouthPixels = static_cast<double>(cv::countNonZero(mouthMask));
        if (mouthPixels < 18.0) {
            return 0.0;
        }

        cv::Mat hsv;
        cv::Mat lab;
        cv::cvtColor(frame(bounds), hsv, cv::COLOR_BGR2HSV);
        cv::cvtColor(frame(bounds), lab, cv::COLOR_BGR2Lab);

        cv::Mat lowRed;
        cv::Mat highRed;
        cv::Mat hsvRed;
        cv::inRange(hsv, cv::Scalar(0, 42, 48), cv::Scalar(20, 255, 255), lowRed);
        cv::inRange(hsv, cv::Scalar(155, 42, 48), cv::Scalar(179, 255, 255), highRed);
        cv::bitwise_or(lowRed, highRed, hsvRed);

        cv::Mat labRed;
        cv::inRange(lab, cv::Scalar(35, 142, 0), cv::Scalar(255, 255, 255), labRed);
        cv::Mat candidates;
        cv::bitwise_and(hsvRed, labRed, candidates);
        cv::bitwise_and(candidates, mouthMask, candidates);

        cv::Mat openKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
        cv::morphologyEx(candidates, candidates, cv::MORPH_OPEN, openKernel);
        cv::morphologyEx(candidates, candidates, cv::MORPH_CLOSE, openKernel);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(candidates, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        double bestScore = 0.0;
        cv::Rect bestBox;
        for (const std::vector<cv::Point>& contour : contours) {
            double area = cv::contourArea(contour);
            if (area < 5.0) {
                continue;
            }
            cv::Rect box = cv::boundingRect(contour);
            cv::Moments moments = cv::moments(contour);
            if (moments.m00 <= 0.0 || box.width < 3 || box.height < 2) {
                continue;
            }
            double centerX = (moments.m10 / moments.m00) / std::max(1.0, static_cast<double>(bounds.width));
            double centerY = (moments.m01 / moments.m00) / std::max(1.0, static_cast<double>(bounds.height));
            if (centerX < 0.12 || centerX > 0.88 || centerY < 0.16 || centerY > 0.98) {
                continue;
            }

            std::vector<cv::Point> hull;
            cv::convexHull(contour, hull);
            double hullArea = std::max(1.0, cv::contourArea(hull));
            double solidity = clamp01(area / hullArea);
            double areaRatio = area / mouthPixels;
            double areaScore = clamp01((areaRatio - 0.055) / 0.24);
            double shapeScore = clamp01((solidity - 0.35) / 0.55);
            double centerScore = 1.0 - clamp01(std::abs(centerX - 0.5) / 0.40);
            double lowerScore = clamp01((centerY - 0.16) / 0.50);
            double openSupport = clamp01((mouthOpenScore - 0.14) / 0.36);
            double score = (areaScore * 0.55 + shapeScore * 0.20
                            + centerScore * 0.15 + lowerScore * 0.10)
                    * (0.55 + openSupport * 0.45);
            if (areaScore > 0.0 && score > bestScore) {
                bestScore = score;
                bestBox = box;
            }
        }

        if (bestScore > 0.0) {
            lastTongueBox_ = cv::Rect(bounds.x + bestBox.x, bounds.y + bestBox.y,
                                      bestBox.width, bestBox.height);
        }
        return clamp01(bestScore);
    }

    static void dampenForPose(Scores& scores, const HeadPose& pose) {
        if (!pose.valid) {
            return;
        }
        double frontal = clamp01(1.0 - std::max(0.0, std::abs(pose.yaw) - 12.0) / 28.0);
        scores.smile *= frontal;
        scores.sad *= frontal;
        scores.tongue *= frontal;
        scores.wink *= frontal;
        scores.leftClosed *= frontal;
        scores.rightClosed *= frontal;
        scores.leftNarrow *= frontal;
        scores.rightNarrow *= frontal;
        scores.browRaiseLeft *= frontal;
        scores.browRaiseRight *= frontal;
        scores.browSadLeft *= frontal;
        scores.browSadRight *= frontal;
        scores.browFurrow *= frontal;
        scores.blink = std::min(scores.leftClosed, scores.rightClosed);
    }

    Scores smoothScores(const Scores& raw) {
        auto smooth = [](double oldValue, double newValue, double rise, double fall) {
            return clamp01(asymmetricSmooth(oldValue, newValue, rise, fall));
        };
        Scores next;
        next.smile = smooth(smoothedScores_.smile, raw.smile, 0.52, 0.28);
        next.mouthOpen = smooth(smoothedScores_.mouthOpen, raw.mouthOpen, 0.62, 0.34);
        next.mouthWide = smooth(smoothedScores_.mouthWide, raw.mouthWide, 0.52, 0.28);
        next.sad = smooth(smoothedScores_.sad, raw.sad, 0.48, 0.25);
        next.tongue = smooth(smoothedScores_.tongue, raw.tongue, 0.76, 0.38);
        next.leftClosed = smooth(smoothedScores_.leftClosed, raw.leftClosed, 0.88, 0.58);
        next.rightClosed = smooth(smoothedScores_.rightClosed, raw.rightClosed, 0.88, 0.58);
        next.leftNarrow = smooth(smoothedScores_.leftNarrow, raw.leftNarrow, 0.58, 0.34);
        next.rightNarrow = smooth(smoothedScores_.rightNarrow, raw.rightNarrow, 0.58, 0.34);
        next.blink = std::min(next.leftClosed, next.rightClosed);
        next.wink = std::max(next.leftClosed * (1.0 - next.rightClosed),
                             next.rightClosed * (1.0 - next.leftClosed));
        next.mouthActivity = smooth(smoothedScores_.mouthActivity, raw.mouthActivity, 0.68, 0.34);
        next.browRaiseLeft = smooth(smoothedScores_.browRaiseLeft, raw.browRaiseLeft, 0.48, 0.26);
        next.browRaiseRight = smooth(smoothedScores_.browRaiseRight, raw.browRaiseRight, 0.48, 0.26);
        next.browSadLeft = smooth(smoothedScores_.browSadLeft, raw.browSadLeft, 0.48, 0.26);
        next.browSadRight = smooth(smoothedScores_.browSadRight, raw.browSadRight, 0.48, 0.26);
        next.browFurrow = smooth(smoothedScores_.browFurrow, raw.browFurrow, 0.52, 0.28);
        smoothedScores_ = next;
        return next;
    }

    FaceParts classify(const Scores& scores) {
        MouthState mouth = MouthState::Neutral;
        if (scores.tongue >= (previousParts_.mouth == MouthState::Funny ? 0.34 : 0.52)) {
            mouth = MouthState::Funny;
        } else if (scores.mouthOpen >= (previousParts_.mouth == MouthState::Surprised ? 0.48 : 0.66)
                && scores.mouthActivity < 0.58) {
            mouth = MouthState::Surprised;
        } else if (scores.mouthActivity >= (previousParts_.mouth == MouthState::Talking ? 0.18 : 0.32)
                && scores.mouthOpen > 0.10) {
            mouth = MouthState::Talking;
        } else if (scores.smile >= (previousParts_.mouth == MouthState::Happy ? 0.32 : 0.52)) {
            mouth = MouthState::Happy;
        } else if (scores.sad >= (previousParts_.mouth == MouthState::Sad ? 0.34 : 0.56)) {
            mouth = MouthState::Sad;
        }

        auto classifyEye = [&](double closed, double narrow, EyeState previous) {
            if (closed >= (previous == EyeState::Closed ? 0.38 : 0.66)) {
                return EyeState::Closed;
            }
            if (narrow >= (previous == EyeState::Focused ? 0.38 : 0.58)
                    && smoothedScores_.browFurrow > 0.42) {
                return EyeState::Focused;
            }
            return EyeState::Open;
        };
        auto classifyBrow = [](double raised, double sad, double furrow, BrowState previous) {
            if (raised >= (previous == BrowState::Raised ? 0.30 : 0.52)) return BrowState::Raised;
            if (furrow >= (previous == BrowState::Focused ? 0.32 : 0.54)) return BrowState::Focused;
            if (sad >= (previous == BrowState::Sad ? 0.30 : 0.52)) return BrowState::Sad;
            return BrowState::Neutral;
        };

        FaceParts requested;
        requested.mouth = mouth;
        requested.leftEye = classifyEye(scores.leftClosed, scores.leftNarrow, previousParts_.leftEye);
        requested.rightEye = classifyEye(scores.rightClosed, scores.rightNarrow, previousParts_.rightEye);
        requested.leftBrow = classifyBrow(scores.browRaiseLeft, scores.browSadLeft,
                                          scores.browFurrow, previousParts_.leftBrow);
        requested.rightBrow = classifyBrow(scores.browRaiseRight, scores.browSadRight,
                                           scores.browFurrow, previousParts_.rightBrow);

        FaceParts stable;
        stable.mouth = mouthChoice_.update(requested.mouth,
                requested.mouth == MouthState::Talking ? 1 : 2);
        stable.leftEye = leftEyeChoice_.update(requested.leftEye,
                requested.leftEye == EyeState::Closed ? 1 : 2);
        stable.rightEye = rightEyeChoice_.update(requested.rightEye,
                requested.rightEye == EyeState::Closed ? 1 : 2);
        stable.leftBrow = leftBrowChoice_.update(requested.leftBrow, 2);
        stable.rightBrow = rightBrowChoice_.update(requested.rightBrow, 2);
        previousParts_ = stable;
        return stable;
    }

    void resetSignals() {
        smoothedScores_ = {};
        lastTongueBox_ = {};
        geometryInitialized_ = false;
        resetStates();
    }

    void resetStates() {
        previousParts_ = {};
        mouthChoice_.reset(MouthState::Neutral);
        leftEyeChoice_.reset(EyeState::Open);
        rightEyeChoice_.reset(EyeState::Open);
        leftBrowChoice_.reset(BrowState::Neutral);
        rightBrowChoice_.reset(BrowState::Neutral);
    }

    void annotate(cv::Mat& frame, const TrackingResult& result,
                  const std::vector<cv::Point2f>& landmarks) const {
        if (result.faceFound) {
            cv::rectangle(frame, result.face, cv::Scalar(60, 205, 255), 2, cv::LINE_AA);
        }
        if (lastTongueBox_.area() > 0 && result.scores.tongue > 0.12) {
            cv::rectangle(frame, lastTongueBox_,
                          result.parts.mouth == MouthState::Funny
                                  ? cv::Scalar(70, 90, 255) : cv::Scalar(210, 150, 255),
                          result.parts.mouth == MouthState::Funny ? 2 : 1, cv::LINE_AA);
        }
        if (config_.drawLandmarks && landmarks.size() == kLandmarkCount) {
            drawLandmarkGroup(frame, landmarks, 0, 16, false, cv::Scalar(255, 175, 72));
            drawLandmarkGroup(frame, landmarks, 17, 21, false, cv::Scalar(88, 235, 255));
            drawLandmarkGroup(frame, landmarks, 22, 26, false, cv::Scalar(88, 235, 255));
            drawLandmarkGroup(frame, landmarks, 27, 35, false, cv::Scalar(224, 162, 255));
            drawLandmarkGroup(frame, landmarks, 36, 41, true, cv::Scalar(104, 228, 136));
            drawLandmarkGroup(frame, landmarks, 42, 47, true, cv::Scalar(104, 228, 136));
            drawLandmarkGroup(frame, landmarks, 48, 59, true, cv::Scalar(96, 120, 255));
            drawLandmarkGroup(frame, landmarks, 60, 67, true, cv::Scalar(96, 120, 255));
        }

        int panelWidth = std::min(760, std::max(1, frame.cols - 24));
        cv::rectangle(frame, cv::Rect(12, 12, panelWidth, 96), cv::Scalar(28, 31, 36), -1);
        cv::Scalar primaryColor = result.faceFound ? cv::Scalar(80, 220, 120) : cv::Scalar(70, 120, 255);
        std::string primary = result.faceFound
                ? std::string("Mouth ") + wire(result.parts.mouth)
                  + "  Eyes " + wire(result.parts.leftEye) + "/" + wire(result.parts.rightEye)
                  + "  Brows " + wire(result.parts.leftBrow) + "/" + wire(result.parts.rightBrow)
                : "No face";
        std::string secondary = result.status + "  q " + fixed(result.trackingQuality)
                + "  det " + fixed(result.detectionConfidence)
                + "  lm " + fixed(result.landmarkConfidence)
                + "  tongue " + fixed(result.scores.tongue)
                + "  " + fixed(result.fps, 1) + " fps";
        std::string pose = result.pose.valid
                ? "pose pitch " + fixed(result.pose.pitch, 1)
                  + " yaw " + fixed(result.pose.yaw, 1)
                  + " roll " + fixed(result.pose.roll, 1)
                : "pose unavailable";
        cv::putText(frame, primary, cv::Point(24, 42), cv::FONT_HERSHEY_SIMPLEX,
                    0.54, primaryColor, 2, cv::LINE_AA);
        cv::putText(frame, secondary, cv::Point(24, 68), cv::FONT_HERSHEY_SIMPLEX,
                    0.44, cv::Scalar(235, 238, 242), 1, cv::LINE_AA);
        cv::putText(frame, pose + "  [C] recalibrate  [Q] quit", cv::Point(24, 94),
                    cv::FONT_HERSHEY_SIMPLEX, 0.42, cv::Scalar(190, 198, 208), 1, cv::LINE_AA);
    }

    static void drawLandmarkGroup(cv::Mat& frame, const std::vector<cv::Point2f>& landmarks,
                                  int first, int last, bool closed, const cv::Scalar& color) {
        std::vector<cv::Point> points;
        points.reserve(static_cast<std::size_t>(last - first + 1));
        for (int index = first; index <= last; ++index) {
            const cv::Point2f& point = landmarks[static_cast<std::size_t>(index)];
            points.emplace_back(static_cast<int>(std::lround(point.x)),
                                static_cast<int>(std::lround(point.y)));
        }
        cv::polylines(frame, points, closed, color, 1, cv::LINE_AA);
    }

    TrackerConfig config_;
    cv::Ptr<cv::FaceDetectorYN> detector_;
    cv::dnn::Net landmarkNet_;
    Calibrator calibrator_;
    cv::Rect lastFace_;
    cv::Rect lastLandmarkFace_;
    std::vector<cv::Point2f> lastLandmarks_;
    bool lastFaceValid_ = false;
    int missedDetections_ = 99;
    int missingFaceFrames_ = 0;
    int badLandmarkFrames_ = 0;
    bool diagnosticReported_ = false;
    bool calibrationDiagnosticReported_ = false;
    cv::Rect lastTongueBox_;
    Geometry previousGeometry_;
    bool geometryInitialized_ = false;
    Scores smoothedScores_;
    FaceParts previousParts_;
    StableChoice<MouthState> mouthChoice_{MouthState::Neutral};
    StableChoice<EyeState> leftEyeChoice_{EyeState::Open};
    StableChoice<EyeState> rightEyeChoice_{EyeState::Open};
    StableChoice<BrowState> leftBrowChoice_{BrowState::Neutral};
    StableChoice<BrowState> rightBrowChoice_{BrowState::Neutral};
};

NeuralFaceTracker::NeuralFaceTracker(TrackerConfig config)
        : impl_(std::make_unique<Impl>(std::move(config))) {
}

NeuralFaceTracker::~NeuralFaceTracker() = default;

TrackingResult NeuralFaceTracker::process(cv::Mat& frame, double fps) {
    return impl_->process(frame, fps);
}

void NeuralFaceTracker::recalibrate() {
    impl_->recalibrate();
}

void NeuralFaceTracker::checkModels() {
    impl_->checkModels();
}

}  // namespace facetrack
