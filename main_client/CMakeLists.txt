cmake_minimum_required(VERSION 3.20)

project(facetrack_client VERSION 2.0.0 LANGUAGES CXX)

find_package(OpenCV 4.5 REQUIRED COMPONENTS
    core
    imgproc
    dnn
    objdetect
    videoio
    highgui
    calib3d
)

add_executable(facetrack_client
    src/main.cpp
    src/tracker.cpp
    src/udp_bridge.cpp
)

target_compile_features(facetrack_client PRIVATE cxx_std_20)
target_include_directories(facetrack_client PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(facetrack_client PRIVATE ${OpenCV_LIBS})

if(WIN32)
    target_link_libraries(facetrack_client PRIVATE ws2_32)
endif()

if(MSVC)
    target_compile_options(facetrack_client PRIVATE /W4 /permissive-)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(facetrack_client PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wconversion
        -Wshadow
    )
endif()

set(FACETRACK_FACE_MODEL_MODERN
    "${CMAKE_CURRENT_SOURCE_DIR}/models/face_detection_yunet_2023mar.onnx")
set(FACETRACK_FACE_MODEL_LEGACY
    "${CMAKE_CURRENT_SOURCE_DIR}/models/face_detection_yunet_2022mar.onnx")
set(FACETRACK_LANDMARK_MODEL
    "${CMAKE_CURRENT_SOURCE_DIR}/models/fan2_68_landmark.onnx")

if(EXISTS "${FACETRACK_FACE_MODEL_MODERN}"
        AND EXISTS "${FACETRACK_FACE_MODEL_LEGACY}"
        AND EXISTS "${FACETRACK_LANDMARK_MODEL}")
    add_custom_target(facetrack_models ALL
        COMMAND ${CMAKE_COMMAND} -E make_directory
            "$<TARGET_FILE_DIR:facetrack_client>/models"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${FACETRACK_FACE_MODEL_MODERN}"
            "$<TARGET_FILE_DIR:facetrack_client>/models/face_detection_yunet_2023mar.onnx"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${FACETRACK_FACE_MODEL_LEGACY}"
            "$<TARGET_FILE_DIR:facetrack_client>/models/face_detection_yunet_2022mar.onnx"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${FACETRACK_LANDMARK_MODEL}"
            "$<TARGET_FILE_DIR:facetrack_client>/models/fan2_68_landmark.onnx"
        COMMENT "Packaging YuNet and FAN models beside the executable"
        VERBATIM
    )
    add_dependencies(facetrack_client facetrack_models)
endif()

install(TARGETS facetrack_client RUNTIME DESTINATION .)
if(EXISTS "${FACETRACK_FACE_MODEL_MODERN}"
        AND EXISTS "${FACETRACK_FACE_MODEL_LEGACY}"
        AND EXISTS "${FACETRACK_LANDMARK_MODEL}")
    install(FILES
        "${FACETRACK_FACE_MODEL_MODERN}"
        "${FACETRACK_FACE_MODEL_LEGACY}"
        "${FACETRACK_LANDMARK_MODEL}"
        DESTINATION models
    )
endif()
