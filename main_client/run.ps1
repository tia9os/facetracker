$ErrorActionPreference = "Stop"

$ClientDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BuildDir = Join-Path $ClientDir "build"
$Cache = Join-Path $BuildDir "CMakeCache.txt"

if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
    throw "CMake is not installed or is not on PATH."
}

if (-not (Test-Path $Cache)) {
    $ConfigureArgs = @(
        "-S", $ClientDir,
        "-B", $BuildDir,
        "-DCMAKE_BUILD_TYPE=Release"
    )

    if ($env:VCPKG_ROOT) {
        $Toolchain = Join-Path $env:VCPKG_ROOT "scripts/buildsystems/vcpkg.cmake"
        if (Test-Path $Toolchain) {
            $ConfigureArgs += "-DCMAKE_TOOLCHAIN_FILE=$Toolchain"
            $ConfigureArgs += "-DVCPKG_TARGET_TRIPLET=x64-windows"
        }
    }

    & cmake @ConfigureArgs
    if ($LASTEXITCODE -ne 0) {
        throw "CMake configuration failed."
    }
}

& cmake --build $BuildDir --config Release --parallel
if ($LASTEXITCODE -ne 0) {
    throw "Build failed."
}

$Candidates = @(
    (Join-Path $BuildDir "Release/facetrack_client.exe"),
    (Join-Path $BuildDir "facetrack_client.exe")
)
$Binary = $Candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $Binary) {
    throw "Built executable was not found."
}

& $Binary @args
exit $LASTEXITCODE
