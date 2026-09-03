// Baseline: same pi algorithm compiled straight to a native exe (no RTorch).
// Compares against the RTorch path (DLL load + rtorch_main) to isolate
// framework dispatch overhead.
#include <chrono>
#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
    long long terms = 200000000LL;
    if (argc > 1) {
        long long t = std::atoll(argv[1]);
        if (t > 0) terms = t;
    }
    std::printf("[native] pi benchmark: %lld terms per arctan series\n", terms);

    auto t0 = std::chrono::high_resolution_clock::now();
    auto atan_inv = [&](double x) -> double {
        double sum = 0.0, term = 1.0 / x, x2 = x * x;
        for (long long k = 0; k < terms; ++k) {
            double sign = (k & 1) ? -1.0 : 1.0;
            sum += sign * term / (2.0 * (double)k + 1.0);
            term /= x2;
        }
        return sum;
    };
    double pi = 16.0 * atan_inv(5.0) - 4.0 * atan_inv(239.0);
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("[native] pi = %.15f\n", pi);
    std::printf("[native] elapsed = %.3f ms  (total %lld series terms)\n", ms, 2 * terms);
    std::printf("[native] throughput = %.3fM terms/s\n", (double)(2 * terms) / (ms / 1000.0) / 1e6);
    return 0;
}
