#pragma once

#include <memory>
#include <string>

#include "types.hpp"

namespace facetrack {

class UdpBridge {
public:
    UdpBridge(const std::string& host, int port, int targetFps);
    ~UdpBridge();

    UdpBridge(const UdpBridge&) = delete;
    UdpBridge& operator=(const UdpBridge&) = delete;

    void send(const TrackingResult& result);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace facetrack
