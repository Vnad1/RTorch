// RTorch demo formula — the compute file (first argument to `rtorch`).
// Contract: must export `extern "C" int rtorch_main(int, char**)`.
// Uses the helper "副框架" compiled in via `with` (helper.cpp).

#include "helper.hpp"

#include <vector>
#include <chrono>
#include <cstdio>
#include <cstdlib>

extern "C" __declspec(dllexport)
int rtorch_main(int argc, char** argv) {
    (void)argc;
    std::printf("[rtorch] formula rtorch_main called\n");
    for (int i = 0; i < argc; ++i) {
        std::printf("[rtorch]   argv[%d] = %s\n", i, argv[i]);
    }

    const std::size_t n = 512;
    const std::size_t total = n * n;

    std::vector<float> a(total), b(total), c(total, 0.0f);
    for (std::size_t i = 0; i < total; ++i) {
        a[i] = (float)(i % 251) / 251.0f;
        b[i] = (float)((i * 7) % 251) / 251.0f;
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    rtorch_demo::matmul(a.data(), b.data(), c.data(), n);
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    float checksum = 0.0f;
    for (std::size_t i = 0; i < total; ++i) {
        checksum += c[i];
    }
    std::printf("[rtorch] matmul n=%zu done in %.3f ms  checksum=%.6f\n", n, ms, checksum);
    return 0;
}
