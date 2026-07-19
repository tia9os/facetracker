#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cctype>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include "tracker.hpp"
#include "udp_bridge.hpp"

namespace fs = std::filesystem;

namespace {

volatile std::sig_atomic_t stopRequested = 0;

void requestStop(int) {
    stopRequested = 1;
}

struct Options {
    int camera = 0;
    int width = 1280;
    int height = 720;
    int fps = 30;
    std::string format = "MJPG";
    std::string host = "127.0.0.1";
    int port = 34321;
    int calibrationFrames = 45;
    bool mirror = false;
    bool headless = false;
    bool drawLandmarks = true;
    bool diagnostics = false;
    bool enableTongue = true;
    bool checkModels = false;
    fs::path faceModel;
    fs::path landmarkModel;
    fs::path executableDir;
};

fs::path executableDirectory(const char* argumentZero) {
    std::error_code error;
    fs::path path = fs::read_symlink("/proc/self/exe", error);
    if (!error && !path.empty()) {
        return path.parent_path();
    }
    path = argumentZero == nullptr ? fs::current_path() : fs::path(argumentZero);
    if (path.is_relative()) {
        path = fs::absolute(path, error);
    }
    return !error && path.has_parent_path() ? path.parent_path() : fs::current_path();
}

std::string requireValue(int& index, int count, char** arguments, std::string_view option) {
    if (index + 1 >= count) {
        throw std::runtime_error(std::string(option) + " requires a value");
    }
    return arguments[++index];
}

int parseInteger(const std::string& text, std::string_view option) {
    int value = 0;
    auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc() || end != text.data() + text.size()) {
        throw std::runtime_error(std::string(option) + " requires a whole number, got: " + text);
    }
    return value;
}

void printUsage() {
    std::cout
            << "FaceTrack Client 2 - neural calibrated tracking\n"
            << "Usage: facetrack_linux_client2 [options]\n\n"
            << "  --camera N              native camera index (default 0)\n"
            << "  --width N               capture width (default 1280)\n"
            << "  --height N              capture height (default 720)\n"
            << "  --fps N                 capture/UDP target FPS (default 30)\n"
            << "  --format FOURCC         camera format (default MJPG; use default for auto)\n"
            << "  --mirror                mirror input and preview\n"
            << "  --headless              disable preview window\n"
            << "  --host IPv4             Minecraft host (default 127.0.0.1)\n"
            << "  --port N                Minecraft UDP port (default 34321)\n"
            << "  --calibration-frames N  neutral samples, 15-300 (default 45)\n"
            << "  --face-model PATH       YuNet ONNX model\n"
            << "  --landmark-model PATH   FAN 68-point ONNX model\n"
            << "  --no-landmarks          hide landmark contours\n"
            << "  --diagnostics           print neural measurement diagnostics\n"
            << "  --disable-tongue        disable inner-mouth tongue detection\n"
            << "  --check-models          validate neural models and exit\n"
            << "  --help                  show this help\n";
}

Options parseOptions(int argc, char** argv) {
    Options options;
    options.executableDir = executableDirectory(argc > 0 ? argv[0] : nullptr);
    for (int index = 1; index < argc; ++index) {
        std::string argument = argv[index];
        if (argument == "--help" || argument == "-h") {
            printUsage();
            std::exit(0);
        } else if (argument == "--camera") {
            options.camera = parseInteger(requireValue(index, argc, argv, argument), argument);
        } else if (argument == "--width") {
            options.width = parseInteger(requireValue(index, argc, argv, argument), argument);
        } else if (argument == "--height") {
            options.height = parseInteger(requireValue(index, argc, argv, argument), argument);
        } else if (argument == "--fps") {
            options.fps = parseInteger(requireValue(index, argc, argv, argument), argument);
        } else if (argument == "--format") {
            options.format = requireValue(index, argc, argv, argument);
        } else if (argument == "--host") {
            options.host = requireValue(index, argc, argv, argument);
        } else if (argument == "--port") {
            options.port = parseInteger(requireValue(index, argc, argv, argument), argument);
        } else if (argument == "--calibration-frames") {
            options.calibrationFrames = parseInteger(requireValue(index, argc, argv, argument), argument);
        } else if (argument == "--face-model") {
            options.faceModel = requireValue(index, argc, argv, argument);
        } else if (argument == "--landmark-model") {
            options.landmarkModel = requireValue(index, argc, argv, argument);
        } else if (argument == "--mirror") {
            options.mirror = true;
        } else if (argument == "--headless") {
            options.headless = true;
        } else if (argument == "--no-landmarks") {
            options.drawLandmarks = false;
        } else if (argument == "--diagnostics") {
            options.diagnostics = true;
        } else if (argument == "--disable-tongue") {
            options.enableTongue = false;
        } else if (argument == "--check-models") {
            options.checkModels = true;
        } else {
            throw std::runtime_error("Unknown option: " + argument);
        }
    }

    if (options.camera < 0) throw std::runtime_error("--camera must be zero or greater");
    if (options.width < 320 || options.width > 7680) throw std::runtime_error("--width must be 320-7680");
    if (options.height < 240 || options.height > 4320) throw std::runtime_error("--height must be 240-4320");
    if (options.fps < 1 || options.fps > 120) throw std::runtime_error("--fps must be 1-120");
    if (options.port < 1 || options.port > 65535) throw std::runtime_error("--port must be 1-65535");
    if (options.calibrationFrames < 15 || options.calibrationFrames > 300) {
        throw std::runtime_error("--calibration-frames must be 15-300");
    }
    if (options.host.empty()) throw std::runtime_error("--host cannot be empty");
#if defined(__linux__)
    if (std::getenv("DISPLAY") == nullptr && std::getenv("WAYLAND_DISPLAY") == nullptr) {
        options.headless = true;
    }
#endif
    return options;
}

std::optional<fs::path> firstFile(const std::vector<fs::path>& candidates) {
    for (const fs::path& candidate : candidates) {
        std::error_code error;
        if (fs::is_regular_file(candidate, error)) {
            fs::path canonical = fs::weakly_canonical(candidate, error);
            return error ? candidate : canonical;
        }
    }
    return std::nullopt;
}

fs::path resolveFaceModel(const Options& options) {
    if (!options.faceModel.empty()) {
        return options.faceModel;
    }
    auto model = firstFile({
            options.executableDir / "models/face_detection_yunet_2023mar.onnx",
            options.executableDir / "../models/face_detection_yunet_2023mar.onnx",
            options.executableDir / "../../linux_client1/models/face_detection_yunet_2023mar.onnx",
            options.executableDir / "../../../linux_client1/models/face_detection_yunet_2023mar.onnx",
            fs::path("linux_client2/models/face_detection_yunet_2023mar.onnx"),
            fs::path("linux_client1/models/face_detection_yunet_2023mar.onnx"),
            fs::path("../linux_client1/models/face_detection_yunet_2023mar.onnx"),
    });
    if (!model) {
        throw std::runtime_error(
                "YuNet model not found. Use --face-model or restore linux_client2/models."
        );
    }
    return *model;
}

fs::path resolveLandmarkModel(const Options& options) {
    if (!options.landmarkModel.empty()) {
        return options.landmarkModel;
    }
    auto model = firstFile({
            options.executableDir / "models/fan2_68_landmark.onnx",
            options.executableDir / "../models/fan2_68_landmark.onnx",
            options.executableDir / "../../client4/models/fan2_68_landmark.onnx",
            options.executableDir / "../../client3/models/fan2_68_landmark.onnx",
            options.executableDir / "../../../client4/models/fan2_68_landmark.onnx",
            fs::path("linux_client2/models/fan2_68_landmark.onnx"),
            fs::path("client4/models/fan2_68_landmark.onnx"),
            fs::path("client3/models/fan2_68_landmark.onnx"),
            fs::path("../client4/models/fan2_68_landmark.onnx"),
    });
    if (!model) {
        throw std::runtime_error(
                "FAN landmark model not found. Use --landmark-model or restore linux_client2/models."
        );
    }
    return *model;
}

int cameraFourcc(std::string format) {
    std::transform(format.begin(), format.end(), format.begin(), [](unsigned char character) {
        return static_cast<char>(std::toupper(character));
    });
    if (format == "AUTO" || format == "DEFAULT" || format.empty()) {
        return 0;
    }
    if (format == "MJPEG") {
        format = "MJPG";
    }
    if (format.size() != 4) {
        throw std::runtime_error("--format must be a four-character FOURCC or default");
    }
    return cv::VideoWriter::fourcc(format[0], format[1], format[2], format[3]);
}

int preferredCameraBackend() {
#if defined(_WIN32)
    return cv::CAP_MSMF;
#elif defined(__APPLE__)
    return cv::CAP_AVFOUNDATION;
#elif defined(__linux__)
    return cv::CAP_V4L2;
#else
    return cv::CAP_ANY;
#endif
}

const char* platformName() {
#if defined(_WIN32)
    return "Windows";
#elif defined(__APPLE__)
    return "macOS";
#elif defined(__linux__)
    return "Linux";
#else
    return "Unknown OS";
#endif
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Options options = parseOptions(argc, argv);
        int fourcc = cameraFourcc(options.format);
        facetrack::TrackerConfig trackerConfig;
        trackerConfig.faceModel = resolveFaceModel(options);
        trackerConfig.landmarkModel = resolveLandmarkModel(options);
        trackerConfig.calibrationFrames = options.calibrationFrames;
        trackerConfig.drawLandmarks = options.drawLandmarks;
        trackerConfig.diagnostics = options.diagnostics;
        trackerConfig.enableTongue = options.enableTongue;

        cv::setUseOptimized(true);
        cv::setNumThreads(static_cast<int>(std::max(1U, std::thread::hardware_concurrency())));
        facetrack::NeuralFaceTracker tracker(std::move(trackerConfig));
        if (options.checkModels) {
            tracker.checkModels();
            std::cout << "YuNet and FAN models passed inference validation.\n";
            return 0;
        }

        facetrack::UdpBridge udp(options.host, options.port, options.fps);
        cv::VideoCapture capture(options.camera, preferredCameraBackend());
        if (!capture.isOpened()) {
            capture.open(options.camera, cv::CAP_ANY);
        }
        if (!capture.isOpened()) {
            throw std::runtime_error("Unable to open camera " + std::to_string(options.camera));
        }
        if (fourcc != 0 && !capture.set(cv::CAP_PROP_FOURCC, fourcc)) {
            std::cerr << "Warning: camera rejected format " << options.format << '\n';
        }
        capture.set(cv::CAP_PROP_FRAME_WIDTH, options.width);
        capture.set(cv::CAP_PROP_FRAME_HEIGHT, options.height);
        capture.set(cv::CAP_PROP_FPS, options.fps);
        capture.set(cv::CAP_PROP_BUFFERSIZE, 1);

        int actualWidth = static_cast<int>(std::lround(capture.get(cv::CAP_PROP_FRAME_WIDTH)));
        int actualHeight = static_cast<int>(std::lround(capture.get(cv::CAP_PROP_FRAME_HEIGHT)));
        double actualFps = capture.get(cv::CAP_PROP_FPS);
        std::cerr << "FaceTrack Client 2 on " << platformName() << '\n'
                  << "Neural pipeline: YuNet detection + FAN 68 landmarks\n"
                  << "Camera " << options.camera << " " << actualWidth << "x" << actualHeight
                  << " @ " << actualFps << " fps, format " << options.format << '\n'
                  << "UDP " << options.host << ':' << options.port << '\n'
                  << "Hold a relaxed neutral face, eyes open, while calibration completes.\n";

        std::signal(SIGINT, requestStop);
        std::signal(SIGTERM, requestStop);
        cv::Mat frame;
        int consecutiveReadFailures = 0;
        std::int64_t countedFrames = 0;
        double measuredFps = 0.0;
        auto measurementStarted = std::chrono::steady_clock::now();
        std::string lastStatus;

        while (stopRequested == 0) {
            if (!capture.read(frame) || frame.empty()) {
                ++consecutiveReadFailures;
                if (consecutiveReadFailures >= 150) {
                    throw std::runtime_error("Camera stopped providing frames");
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                continue;
            }
            consecutiveReadFailures = 0;
            if (options.mirror) {
                cv::flip(frame, frame, 1);
            }

            facetrack::TrackingResult result = tracker.process(frame, measuredFps);
            udp.send(result);
            if (options.headless && result.status != lastStatus) {
                std::cerr << result.status << '\n';
                lastStatus = result.status;
            }

            ++countedFrames;
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - measurementStarted).count();
            if (elapsed >= 0.75) {
                measuredFps = static_cast<double>(countedFrames) / elapsed;
                countedFrames = 0;
                measurementStarted = now;
            }

            if (!options.headless) {
                cv::imshow("FaceTrack Client 2", frame);
                int key = cv::waitKey(1);
                if (key == 'q' || key == 'Q' || key == 27) {
                    break;
                }
                if (key == 'c' || key == 'C') {
                    tracker.recalibrate();
                    std::cerr << "Neutral calibration restarted.\n";
                }
            }
        }
        if (!options.headless) {
            cv::destroyAllWindows();
        }
        std::cerr << "FaceTrack stopped.\n";
        return 0;
    } catch (const cv::Exception& exception) {
        std::cerr << "OpenCV error: " << exception.what() << '\n';
        return 1;
    } catch (const std::exception& exception) {
        std::cerr << "FaceTrack Client 2 error: " << exception.what() << '\n';
        return 1;
    }
}
