#!/usr/bin/env bash
set -euo pipefail

client_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${client_dir}/build"
binary="${build_dir}/facetrack_client"

if ! command -v cmake >/dev/null 2>&1; then
    if [[ "$(uname -s)" == "Darwin" ]]; then
        echo "Missing CMake. Install it with: brew install cmake opencv" >&2
    else
        echo "Missing CMake. Install build-essential, cmake, and libopencv-dev." >&2
    fi
    exit 1
fi

if [[ ! -f "${build_dir}/CMakeCache.txt" ]]; then
    configure_args=(-S "${client_dir}" -B "${build_dir}" -DCMAKE_BUILD_TYPE=Release)
    if [[ "$(uname -s)" == "Darwin" ]] && command -v brew >/dev/null 2>&1; then
        opencv_prefix="$(brew --prefix opencv)"
        configure_args+=("-DOpenCV_DIR=${opencv_prefix}/lib/cmake/opencv4")
    fi
    cmake "${configure_args[@]}"
fi
cmake --build "${build_dir}" --parallel

exec "${binary}" "$@"
