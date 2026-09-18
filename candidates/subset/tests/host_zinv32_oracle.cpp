// Host oracle for tests/gpu_epochs/zinv32.cuh.  It includes the production
// header and supplies a synchronized four-lane model of __shfl_sync.
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

static const char *EXPECTED_ZINV32_FNV64 = "5d8b9116b71c839d";

static thread_local int g_lane = -1;

struct ShuffleCall {
    uint32_t values[4] = {0, 0, 0, 0};
    bool present[4] = {false, false, false, false};
    bool complete = false;
};

struct ShuffleModel {
    std::mutex mu;
    std::condition_variable cv;
    std::vector<ShuffleCall> calls;
    uint64_t call_count[4] = {0, 0, 0, 0};
    bool failed = false;
    std::string fail_msg;
} g_shuffle;

static void fail_shuffle(const std::string &msg) {
    if (!g_shuffle.failed) {
        g_shuffle.failed = true;
        g_shuffle.fail_msg = msg;
    }
    g_shuffle.cv.notify_all();
}

uint32_t zi_x(uint32_t v, int src) {
    if (g_lane < 0 || g_lane >= 4 || src < 0 || src >= 4) {
        std::lock_guard<std::mutex> lk(g_shuffle.mu);
        fail_shuffle("invalid lane or shuffle source");
        return 0;
    }
    std::unique_lock<std::mutex> lk(g_shuffle.mu);
    const uint64_t idx = g_shuffle.call_count[g_lane]++;
    if (idx >= g_shuffle.calls.size()) {
        fail_shuffle("too many shuffle calls");
        return 0;
    }
    ShuffleCall &call = g_shuffle.calls[(size_t)idx];
    call.values[g_lane] = v;
    call.present[g_lane] = true;
    call.complete = call.present[0] && call.present[1] &&
                    call.present[2] && call.present[3];
    if (call.complete) {
        g_shuffle.cv.notify_all();
    } else {
        g_shuffle.cv.wait_for(lk, std::chrono::seconds(5), [&] {
            return call.complete || g_shuffle.failed;
        });
    }
    if (!call.complete) {
        fail_shuffle("shuffle rendezvous timeout/divergence");
        return 0;
    }
    return call.values[src];
}

#include "gpu_epochs/zinv32.cuh"

static std::string fnv64_file(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return "";
    }
    uint64_t h = 1469598103934665603ull;
    char buf[4096];
    while (f.good()) {
        f.read(buf, sizeof(buf));
        std::streamsize n = f.gcount();
        for (std::streamsize i = 0; i < n; i++) {
            h ^= (unsigned char)buf[i];
            h *= 1099511628211ull;
        }
    }
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    out << std::setw(16) << h;
    return out.str();
}

static bool parse_hex256(const std::string &s, uint64_t out[5]) {
    if (s.size() > 64) {
        return false;
    }
    std::string padded(64 - s.size(), '0');
    padded += s;
    for (int i = 0; i < 4; i++) {
        uint64_t word = 0;
        for (int j = 0; j < 16; j++) {
            char c = padded[64 - 16 * (i + 1) + j];
            unsigned v;
            if (c >= '0' && c <= '9') v = (unsigned)(c - '0');
            else if (c >= 'a' && c <= 'f') v = (unsigned)(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v = (unsigned)(c - 'A' + 10);
            else return false;
            word = (word << 4) | v;
        }
        out[i] = word;
    }
    out[4] = 0;
    return true;
}

static bool parse_hex288_words(const std::string &s, uint32_t out[9]) {
    if (s.size() != 72) {
        return false;
    }
    for (int i = 0; i < 9; i++) {
        uint32_t word = 0;
        for (int j = 0; j < 8; j++) {
            char c = s[72 - 8 * (i + 1) + j];
            unsigned v;
            if (c >= '0' && c <= '9') v = (unsigned)(c - '0');
            else if (c >= 'a' && c <= 'f') v = (unsigned)(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v = (unsigned)(c - 'A' + 10);
            else return false;
            word = (word << 4) | v;
        }
        out[i] = word;
    }
    return true;
}

static std::string hex256(const uint64_t in[5]) {
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (int i = 3; i >= 0; i--) {
        out << std::setw(16) << in[i];
    }
    return out.str();
}

static bool run_inverse(const uint64_t in[5], uint64_t out[5]) {
    {
        std::lock_guard<std::mutex> lk(g_shuffle.mu);
        g_shuffle.calls.assign(100000, ShuffleCall{});
        for (int i = 0; i < 4; i++) {
            g_shuffle.call_count[i] = 0;
        }
        g_shuffle.failed = false;
        g_shuffle.fail_msg.clear();
    }
    std::array<std::array<uint64_t, 5>, 4> roots{};
    std::array<std::thread, 4> threads;
    for (int lane = 0; lane < 4; lane++) {
        for (int i = 0; i < 5; i++) {
            roots[(size_t)lane][(size_t)i] = in[i];
        }
        threads[(size_t)lane] = std::thread([&, lane] {
            g_lane = lane;
            zi_inverse_quad(roots[(size_t)lane].data(), lane);
        });
    }
    for (auto &t : threads) {
        t.join();
    }
    if (g_shuffle.failed) {
        std::cerr << "shuffle error: " << g_shuffle.fail_msg << "\n";
        return false;
    }
    for (int lane = 1; lane < 4; lane++) {
        for (int i = 0; i < 5; i++) {
            if (roots[(size_t)lane][(size_t)i] != roots[0][(size_t)i]) {
                std::cerr << "lane output mismatch\n";
                return false;
            }
        }
    }
    for (int i = 0; i < 5; i++) {
        out[i] = roots[0][(size_t)i];
    }
    return true;
}

int main(int argc, char **argv) {
    const char *path_env = std::getenv("ZINV32_HEADER_PATH");
    std::string header_path = path_env ? path_env :
        "candidates/subset/tests/gpu_epochs/zinv32.cuh";
    std::string digest = fnv64_file(header_path);
    if (digest != EXPECTED_ZINV32_FNV64) {
        std::cerr << "fingerprint mismatch for " << header_path << "\n";
        std::cerr << "expected " << EXPECTED_ZINV32_FNV64 << "\n";
        std::cerr << "actual   " << digest << "\n";
        return 3;
    }
    if (argc > 1 && std::strcmp(argv[1], "--fingerprint-only") == 0) {
        std::cout << digest << "\n";
        return 0;
    }
    if (argc > 1 && std::strcmp(argv[1], "--canon") == 0) {
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line.empty()) {
                continue;
            }
            uint32_t words[9];
            if (!parse_hex288_words(line, words)) {
                std::cerr << "bad 288-bit word input: " << line << "\n";
                return 2;
            }
            zi_canon(words);
            uint64_t out[5];
            for (int i = 0; i < 4; i++) {
                out[i] = (uint64_t)words[2 * i] |
                         ((uint64_t)words[2 * i + 1] << 32);
            }
            out[4] = 0;
            std::cout << hex256(out) << "\n";
        }
        return 0;
    }
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) {
            continue;
        }
        uint64_t in[5], out[5];
        if (!parse_hex256(line, in)) {
            std::cerr << "bad hex input: " << line << "\n";
            return 2;
        }
        if (!run_inverse(in, out)) {
            return 4;
        }
        std::cout << hex256(out) << "\n";
    }
    return 0;
}
