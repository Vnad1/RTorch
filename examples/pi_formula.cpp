// RTorch pi benchmark formula.
// Contract: extern "C" __declspec(dllexport) int rtorch_main(int, char**).
// Computes pi via a large fixed iteration-count series accumulation so the
// wall-clock is a clean speed signal. Uses the Machin-derived arctan series:
//   pi = 16*atan(1/5) - 4*atan(1/239), each atan summed term-by-term.

#include <chrono>
#include <cstdio>
#include <cstdlib>

extern "C" __declspec(dllexport)
int rtorch_main(int argc, char** argv) {
    (void)argc;
    long long terms = 200000000LL; // default 200M terms per atan series
    if (argc > 1) {
        long long t = std::atoll(argv[1]);
        if (t > 0) terms = t;
    }
    std::printf("[rtorch] pi benchmark: %lld terms per arctan series\n", terms);

    auto t0 = std::chrono::high_resolution_clock::now();

    // atan(1/x) = sum_{k>=0} (-1)^k / ((2k+1) x^(2k+1))
    auto atan_inv = [&](double x) -> double {
        double sum = 0.0;
        double term = 1.0 / x;
        double x2 = x * x;
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
    double iters_per_s = (double)(2 * terms) / (ms / 1000.0);

    std::printf("[rtorch] pi = %.15f\n", pi);
    std::printf("[rtorch] elapsed = %.3f ms  (total %lld series terms)\n", ms, 2 * terms);
    std::printf("[rtorch] throughput = %.3fM terms/s\n", iters_per_s / 1e6);
    return 0;
}
