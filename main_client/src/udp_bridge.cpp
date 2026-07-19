#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#endif

#include "udp_bridge.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

#ifndef _WIN32
#include <arpa/inet.h>
#include <cerrno>
#include <cstring>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace facetrack {

namespace {

std::string fixed(double value, int digits) {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(digits) << value;
    return stream.str();
}

std::string encode(const TrackingResult& result) {
    std::ostringstream stream;
    stream << "facetrack_parts|1|"
           << result.timestamp << '|'
           << (result.faceFound ? wire(result.expression) : "no_face") << '|'
           << fixed(result.confidence, 4) << '|'
           << result.faceCount << '|'
           << fixed(result.scores.smile, 4) << '|'
           << fixed(result.scores.mouthOpen, 4) << '|'
           << fixed(result.scores.mouthWide, 4) << '|'
           << fixed(result.scores.sad, 4) << '|'
           << fixed(result.scores.blink, 4) << '|'
           << fixed(result.scores.wink, 4) << '|'
           << fixed(result.fps, 2) << '|'
           << wire(result.parts.mouth) << '|'
           << wire(result.parts.leftEye) << '|'
           << wire(result.parts.rightEye) << '|'
           << wire(result.parts.leftBrow) << '|'
           << wire(result.parts.rightBrow);
    return stream.str();
}

}  // namespace

class UdpBridge::Impl {
public:
    Impl(const std::string& host, int port, int targetFps)
            : interval_(std::chrono::nanoseconds(1'000'000'000LL / std::max(1, targetFps))) {
#ifdef _WIN32
        WSADATA data{};
        int startupResult = ::WSAStartup(MAKEWORD(2, 2), &data);
        if (startupResult != 0) {
            throw std::runtime_error("WSAStartup failed with error " + std::to_string(startupResult));
        }
        winsockStarted_ = true;
#endif
        socket_ = ::socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        if (!socketValid()) {
            int error = lastSocketError();
            cleanupPlatform();
            throw std::runtime_error("Unable to create UDP socket (error " + std::to_string(error) + ")");
        }
        address_.sin_family = AF_INET;
        address_.sin_port = htons(static_cast<std::uint16_t>(port));
#ifdef _WIN32
        int addressResult = ::InetPtonA(AF_INET, host.c_str(), &address_.sin_addr);
#else
        int addressResult = ::inet_pton(AF_INET, host.c_str(), &address_.sin_addr);
#endif
        if (addressResult != 1) {
            closeSocket();
            cleanupPlatform();
            throw std::runtime_error("Invalid IPv4 address: " + host);
        }
    }

    ~Impl() {
        closeSocket();
        cleanupPlatform();
    }

    void send(const TrackingResult& result) {
        auto now = std::chrono::steady_clock::now();
        if (lastSent_.time_since_epoch().count() != 0 && now - lastSent_ < interval_) {
            return;
        }
        std::string payload = encode(result);
#ifdef _WIN32
        int sent = ::sendto(socket_, payload.data(), static_cast<int>(payload.size()), 0,
                            reinterpret_cast<const sockaddr*>(&address_),
                            static_cast<int>(sizeof(address_)));
        bool failed = sent == SOCKET_ERROR;
#else
        ssize_t sent = ::sendto(socket_, payload.data(), payload.size(), 0,
                                reinterpret_cast<const sockaddr*>(&address_), sizeof(address_));
        bool failed = sent < 0;
#endif
        if (failed && !errorReported_) {
            std::cerr << "UDP send failed with error " << lastSocketError() << '\n';
            errorReported_ = true;
        } else if (!failed) {
            errorReported_ = false;
        }
        lastSent_ = now;
    }

private:
#ifdef _WIN32
    using Socket = SOCKET;
    static constexpr Socket kInvalidSocket = INVALID_SOCKET;
#else
    using Socket = int;
    static constexpr Socket kInvalidSocket = -1;
#endif

    bool socketValid() const {
        return socket_ != kInvalidSocket;
    }

    static int lastSocketError() {
#ifdef _WIN32
        return ::WSAGetLastError();
#else
        return errno;
#endif
    }

    void closeSocket() {
        if (!socketValid()) {
            return;
        }
#ifdef _WIN32
        ::closesocket(socket_);
#else
        ::close(socket_);
#endif
        socket_ = kInvalidSocket;
    }

    void cleanupPlatform() {
#ifdef _WIN32
        if (winsockStarted_) {
            ::WSACleanup();
            winsockStarted_ = false;
        }
#endif
    }

    Socket socket_ = kInvalidSocket;
    sockaddr_in address_{};
    std::chrono::nanoseconds interval_{};
    std::chrono::steady_clock::time_point lastSent_{};
    bool errorReported_ = false;
#ifdef _WIN32
    bool winsockStarted_ = false;
#endif
};

UdpBridge::UdpBridge(const std::string& host, int port, int targetFps)
        : impl_(std::make_unique<Impl>(host, port, targetFps)) {
}

UdpBridge::~UdpBridge() = default;

void UdpBridge::send(const TrackingResult& result) {
    impl_->send(result);
}

const char* wire(MouthState value) {
    switch (value) {
        case MouthState::Happy: return "happy";
        case MouthState::Sad: return "sad";
        case MouthState::Surprised: return "surprised";
        case MouthState::Talking: return "talking";
        case MouthState::Funny: return "funny";
        default: return "neutral";
    }
}

const char* wire(EyeState value) {
    switch (value) {
        case EyeState::Closed: return "closed";
        case EyeState::Focused: return "focused";
        default: return "open";
    }
}

const char* wire(BrowState value) {
    switch (value) {
        case BrowState::Raised: return "raised";
        case BrowState::Sad: return "sad";
        case BrowState::Focused: return "focused";
        default: return "neutral";
    }
}

const char* wire(Expression value) {
    switch (value) {
        case Expression::Happy: return "happy";
        case Expression::Funny: return "funny";
        case Expression::Sad: return "sad";
        case Expression::Surprised: return "surprised";
        case Expression::Talking: return "talking";
        case Expression::Blinking: return "blinking";
        case Expression::Winking: return "winking";
        case Expression::Focused: return "focused";
        case Expression::NoFace: return "no_face";
        default: return "neutral";
    }
}

Expression expressionFor(const FaceParts& parts) {
    if (parts.mouth == MouthState::Funny) return Expression::Funny;
    if (parts.leftEye == EyeState::Closed && parts.rightEye == EyeState::Closed) return Expression::Blinking;
    if (parts.leftEye == EyeState::Closed || parts.rightEye == EyeState::Closed) return Expression::Winking;
    if (parts.leftEye == EyeState::Focused || parts.rightEye == EyeState::Focused
            || parts.leftBrow == BrowState::Focused || parts.rightBrow == BrowState::Focused) {
        return Expression::Focused;
    }
    if (parts.mouth == MouthState::Talking) return Expression::Talking;
    if (parts.mouth == MouthState::Surprised) return Expression::Surprised;
    if (parts.mouth == MouthState::Happy) return Expression::Happy;
    if (parts.mouth == MouthState::Sad
            || parts.leftBrow == BrowState::Sad || parts.rightBrow == BrowState::Sad) {
        return Expression::Sad;
    }
    return Expression::Neutral;
}

}  // namespace facetrack
