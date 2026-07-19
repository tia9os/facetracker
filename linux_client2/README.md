# FaceTrack Client 2

`linux_client2` is a cross-platform native C++20 face tracker for the FaceTrack Minecraft mod. The same source supports Linux, Windows, and macOS; each operating system needs its own native build.

1. OpenCV YuNet detects and scores faces.
2. A FAN ONNX network predicts 68 landmarks and per-point confidence.
3. `solvePnP` estimates pitch, yaw, and roll.
4. A short neutral calibration measures the user's normal eye, mouth, and eyebrow geometry.
5. The FAN inner-lip contour constrains tongue color and connected-blob analysis.
6. Pose-gated, temporally filtered measurements produce independent mouth, eye, and eyebrow states.
7. The existing `facetrack_parts|1` UDP protocol sends results to Minecraft.

There is no Haar expression detector, LBF landmark fallback, Java runtime, or guessed emotion classifier. Frames with unreliable neural landmarks are held or rejected instead of being converted into strong expressions.

Native platform integrations:

- Linux: V4L2 camera capture and POSIX UDP.
- Windows: Media Foundation camera capture and Winsock UDP.
- macOS: AVFoundation camera capture and POSIX UDP.

## Included models

The client includes the two neural models required at runtime:

- `linux_client2/models/face_detection_yunet_2023mar.onnx`
- `linux_client2/models/fan2_68_landmark.onnx`

CMake copies both models into a `models` directory beside the compiled executable. A built application directory can therefore be moved without retaining the rest of the source repository, provided the matching OpenCV runtime libraries are installed or bundled. Custom paths can also be supplied with `--face-model` and `--landmark-model`.

## Linux requirements

On Debian or Ubuntu:

```bash
sudo apt install build-essential cmake libopencv-dev
```

Run:

```bash
./linux_client2/run.sh --camera 0 --format MJPG --width 1280 --height 720 --fps 30 --mirror
```

## macOS requirements

Install Xcode Command Line Tools, CMake, and OpenCV:

```bash
xcode-select --install
brew install cmake opencv
```

Run:

```bash
./linux_client2/run.sh --camera 0 --width 1280 --height 720 --fps 30 --mirror
```

The first launch should request camera permission. If capture is denied, enable the terminal application under **System Settings → Privacy & Security → Camera**. `MJPG` support varies between macOS cameras; omit `--format MJPG` or use `--format default` when necessary.

## Windows requirements

Install:

- Visual Studio 2022 Build Tools with **Desktop development with C++**;
- CMake;
- OpenCV 4, conveniently through vcpkg.

Example PowerShell setup:

```powershell
git clone https://github.com/microsoft/vcpkg.git C:\vcpkg
C:\vcpkg\bootstrap-vcpkg.bat
C:\vcpkg\vcpkg.exe install opencv4:x64-windows
$env:VCPKG_ROOT = "C:\vcpkg"
```

Run from PowerShell:

```powershell
.\linux_client2\run.ps1 --camera 0 --width 1280 --height 720 --fps 30 --mirror
```

Alternatively, run `linux_client2\run.bat`. The launcher uses the vcpkg toolchain automatically when `VCPKG_ROOT` is defined. Windows may request camera permission; enable desktop camera access under **Settings → Privacy & security → Camera**.

## Validate the neural models

From the repository root on Linux or macOS:

```bash
./linux_client2/run.sh --check-models
```

On Windows:

```powershell
.\linux_client2\run.ps1 --check-models
```

Expected result:

```text
YuNet and FAN models passed inference validation.
```

This performs an actual FAN forward pass, not only a file check.

## Calibration and use

Start Minecraft with the mod, then use the launcher for the current operating system.

```bash
./linux_client2/run.sh --camera 0 --format MJPG --width 1280 --height 720 --fps 30 --mirror
```

At startup:

1. Face the camera directly.
2. Keep both eyes open.
3. Relax the mouth and eyebrows.
4. Hold still until `Calibrating neutral face` reaches 100%.

Press `C` to recalibrate, `Q` or Escape to quit. Calibration is intentionally required: fixed thresholds are not accurate across different faces, glasses, lenses, camera positions, and lighting.

For headless use on Linux:

```bash
./linux_client2/run.sh --headless --camera 0
```

Calibration progress and tracking state are printed to the terminal.

## Accuracy guidance

- Use soft, even light from in front of the face.
- Avoid a bright window behind the head.
- Keep the face at least about 200 pixels tall.
- Recalibrate after changing the camera position, glasses, or lighting.
- Prefer MJPG at 1280x720 if the camera supports it.
- If processing FPS is too low, try 960x540; FAN still analyzes a dedicated 256x256 face crop.

The preview reports:

- `det`: YuNet face-detection confidence.
- `lm`: mean FAN landmark confidence.
- `q`: combined detection, landmark, and pose quality.
- Pitch/yaw/roll in degrees.

Expression output is deliberately damped when the face turns too far away for reliable asymmetric eye or eyebrow measurements.

## Tongue / funny mouth

Tongue detection is enabled by default. It analyzes the neural inner-lip region plus a narrow landmark-derived corridor below the lower lip, allowing both visible and protruding tongues while avoiding most of the surrounding face. It requires:

- a sufficiently open mouth;
- red/pink agreement in HSV and Lab color spaces;
- a coherent connected blob;
- plausible central/lower-mouth position and shape;
- reliable frontal head pose.

When the filtered tongue score crosses its confidence threshold, the client sends `funny` as the mouth part and expression. The preview shows the tongue score and outlines the accepted blob. Disable it with `--disable-tongue` if unusual colored lighting causes false detections.

## Options

```text
--camera N              native camera index
--width N               capture width
--height N              capture height
--fps N                 capture and UDP target FPS
--format FOURCC         MJPG, YUYV, or another camera FOURCC
--mirror                mirror input
--headless              disable preview
--host IPv4             Minecraft host
--port N                Minecraft UDP port
--calibration-frames N  neutral samples, 15-300
--face-model PATH       YuNet ONNX model
--landmark-model PATH   FAN ONNX model
--no-landmarks          hide landmark contours
--diagnostics           print rejected neural geometry details
--disable-tongue        disable tongue/funny-mouth detection
--check-models          validate models and exit
```

## Manual build: Linux or macOS

```bash
cmake -S linux_client2 -B linux_client2/build -DCMAKE_BUILD_TYPE=Release
cmake --build linux_client2/build --parallel
```

On macOS, if CMake cannot find Homebrew OpenCV:

```bash
cmake -S linux_client2 -B linux_client2/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DOpenCV_DIR="$(brew --prefix opencv)/lib/cmake/opencv4"
cmake --build linux_client2/build --parallel
```

## Manual build: Windows

From a Visual Studio Developer PowerShell:

```powershell
cmake -S linux_client2 -B linux_client2/build `
  -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" `
  -DVCPKG_TARGET_TRIPLET=x64-windows
cmake --build linux_client2/build --config Release --parallel
```

The Windows executable is normally written to:

```text
linux_client2/build/Release/facetrack_linux_client2.exe
```
