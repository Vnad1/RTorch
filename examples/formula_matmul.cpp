// RTorch CPU matmul — multi-threaded, cache-blocked, EXPLICIT AVX2 vectorized.
// Uses __m256 FMA intrinsics in the inner loop so the compiler emits AVX2 FMA
// (the plain loop the framework compiles only sees the -O3 auto-vectorizer).
// C[n×n] = A[n×n] @ B[n×n]. Parallel over row blocks; k/j blocked for cache.
#include "rtorch.h"

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <vector>
#include <thread>
#include <algorithm>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#ifndef NT
#define NT 8
#endif
#ifndef BK
#define BK 64
#endif

unsigned long long rtorch_output_size(int n_in, const rtorch_blob* in, int device) {
    (void)device;
    if (n_in < 2) return 0;
    std::size_t nf = in[0].len / 4;
    std::size_t n = (std::size_t)std::floor(std::sqrt((double)nf));
    return (unsigned long long)(n * n * 4);
}

// Inner k*j with AVX2 FMA: broadcast aik, FMA into 8-wide accumulators.
static inline void axpy8(float* dst, const float* src, float a) {
#if defined(__AVX2__)
    __m256 av = _mm256_set1_ps(a);
    __m256 r = _mm256_loadu_ps(dst);
    __m256 s = _mm256_loadu_ps(src);
    r = _mm256_fmadd_ps(av, s, r);
    _mm256_storeu_ps(dst, r);
#else
    for (int j = 0; j < 8; ++j) dst[j] += a * src[j];
#endif
}

static void matmul_block(const float* A, const float* B, float* C, std::size_t n) {
    std::size_t chunk = (n + NT - 1) / NT;
    std::vector<std::thread> ths;
    auto work = [&](std::size_t i0, std::size_t i1) {
        for (std::size_t i = i0; i < i1; ++i) {
            for (std::size_t kb = 0; kb < n; kb += BK) {
                for (std::size_t j = 0; j + 8 <= n; j += 8) {
                    for (std::size_t k = kb; k < std::min(kb + BK, n); ++k) {
                        axpy8(C + i * n + j, B + k * n + j, A[i * n + k]);
                    }
                }
                // tail (n % 8)
                for (std::size_t k = kb; k < std::min(kb + BK, n); ++k)
                    for (std::size_t j = n - (n % 8); j < n; ++j)
                        C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    };
    for (std::size_t t = 0; t < NT; ++t) {
        std::size_t i0 = t * chunk, i1 = std::min(i0 + chunk, n);
        if (i0 >= n) break;
        ths.emplace_back(work, i0, i1);
    }
    for (auto& th : ths) th.join();
}

int rtorch_compute(int n_in, const rtorch_blob* in, rtorch_blob* out, int device) {
    (void)device;
    if (n_in < 2) return 1;
    std::size_t nf = in[0].len / 4;
    std::size_t n = (std::size_t)std::floor(std::sqrt((double)nf));
    const float* A = (const float*)in[0].data;
    const float* B = (const float*)in[1].data;
    float* C = (float*)out->data;
    std::fill(C, C + n * n, 0.0f);
    matmul_block(A, B, C, n);
    out->len = n * n * 4;
    return 0;
}
