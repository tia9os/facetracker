cmake_minimum_required(VERSION 3.20)

project(facetrack_linux_client2 VERSION 2.0.0 LANGUAGES CXX)

find_package(OpenCV 4.5 REQUIRED COMPONENTS
    core
    imgproc
    dnn
    objdetect
    videoio
    highgui
    calib3d
)

add_executable(facetrack_linux_client2
    src/main.cpp
    src/tracker.cpp
    src/udp_bridge.cpp
)

target_compile_features(facetrack_linux_client2 PRIVATE cxx_std_20)
target_include_directories(facetrack_linux_client2 PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(facetrack_linux_client2 PRIVATE ${OpenCV_LIBS})

if(WIN32)
    target_link_libraries(facetrack_linux_client2 PRIVATE ws2_32)
endif()

if(MSVC)
    target_compile_options(facetrack_linux_client2 PRIVATE /W4 /permissive-)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(facetrack_linux_client2 PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wconversion
        -Wshadow
    )
endif()

set(FACETRACK_FACE_MODEL
    "${CMAKE_CURRENT_SOURCE_DIR}/models/face_detection_yunet_2023mar.onnx")
set(FACETRACK_LANDMARK_MODEL
    "${CMAKE_CURRENT_SOURCE_DIR}/models/fan2_68_landmark.onnx")

if(EXISTS "${FACETRACK_FACE_MODEL}" AND EXISTS "${FACETRACK_LANDMARK_MODEL}")
    add_custom_command(TARGET facetrack_linux_client2 POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory
            "$<TARGET_FILE_DIR:facetrack_linux_client2>/models"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${FACETRACK_FACE_MODEL}"
            "$<TARGET_FILE_DIR:facetrack_linux_client2>/models/face_detection_yunet_2023mar.onnx"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${FACETRACK_LANDMARK_MODEL}"
            "$<TARGET_FILE_DIR:facetrack_linux_client2>/models/fan2_68_landmark.onnx"
        COMMENT "Packaging YuNet and FAN models beside the executable"
        VERBATIM
    )
endif()

install(TARGETS facetrack_linux_client2 RUNTIME DESTINATION .)
if(EXISTS "${FACETRACK_FACE_MODEL}" AND EXISTS "${FACETRACK_LANDMARK_MODEL}")
    install(FILES
        "${FACETRACK_FACE_MODEL}"
        "${FACETRACK_LANDMARK_MODEL}"
        DESTINATION models
    )
endif()
