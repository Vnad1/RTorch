// RTorch precision test formula — runs the SAME matmul (C = A*B, fixed 64x64)
// in fp32 / fp16 / fp8(E4M3) / fp4(E2M1) and reports the max abs/rel error vs
// the fp32 reference, so the framework's numerical precision behavior at each
// width is measured. Input: [fp32 A (64*64), fp32 B (64*64)]. Output: 4x2 fp32
// stats (max_abs_err, max_rel_err) for fp32/fp16/fp8/fp4 in that order.
#include "rtorch.h"
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <limits>
#include <cstdio>

#define MM 64
#define NN 64
#define KK 64

// --- IEEE-style quantization (sign, exp_bits, mantissa_bits, implicit 1) ---
// round-to-nearest-even on the mantissa fraction. Returns the quantized float.
static float quantize(float x, int exp_bits, int man_bits) {
    if (x == 0.0f) return 0.0f;
    if (std::isnan(x) || std::isinf(x)) return x;
    bool neg = x < 0.0f;
    float ax = std::fabs(x);
    int max_exp = (1 << exp_bits) - 1;        // biased exp field max (all-ones = inf/nan)
    int  man_scale = 1 << man_bits;           // 2^man_bits
    int  bias = (1 << (exp_bits - 1)) - 1;    // IEEE-like bias

    // clamp to largest finite
    float max_finite = std::ldexp(2.0f - 1.0f / man_scale, max_exp - bias);
    if (ax >= max_finite) { float r = max_finite; return neg ? -r : r; }

    int e = (int)std::floor(std::log2(ax));   // unbiased exponent
    // normalize mantissa fraction into [0,1)
    float frac = ax / std::ldexp(1.0f, e) - 1.0f;   // in [0,1)
    long mi = std::lround(frac * man_scale);         // 0..man_scale
    if (mi >= man_scale) { mi = 0; e += 1; }         // rounded up exactly
    float q = std::ldexp(1.0f + (float)mi / man_scale, e);
    return neg ? -q : q;
}

static float q_fp32(float x){ return x; }
static float q_fp16(float x){ return quantize(x,5,10); }
static float q_fp8 (float x){ return quantize(x,4,3); }   // E4M3
static float q_fp4 (float x){ return quantize(x,2,1); }   // E2M1

typedef float (*QFn)(float);

static float run_matmul_q(const float* A, const float* B, float* C, QFn q) {
    float max_r = 0.0f, max_a = 0.0f;
    for (int i = 0; i < MM; ++i)
        for (int j = 0; j < NN; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < KK; ++k)
                acc += q(A[i*KK+k]) * q(B[k*NN+j]);
            C[i*NN+j] = acc;
            float r = std::fabs(acc);
            if (r > max_a) max_a = r;
        }
    return max_a;
}

// CPU reference (fp32 accumulate, fp32 operands).
static void matmul_ref(const float* A, const float* B, float* C) {
    for (int i = 0; i < MM; ++i)
        for (int j = 0; j < NN; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < KK; ++k) acc += A[i*KK+k] * B[k*NN+j];
            C[i*NN+j] = acc;
        }
}

unsigned long long rtorch_output_size(int n_in, const rtorch_blob* in, int device) {
    (void)device;
    if (n_in < 2) return 0;
    return (unsigned long long)(4 * 2 * sizeof(float));
}

int rtorch_compute(int n_in, const rtorch_blob* in, rtorch_blob* out, int device) {
    (void)device;
    if (n_in < 2 || in[0].len < MM*KK*4 || in[1].len < KK*NN*4) return 1;
    const float* A = (const float*)in[0].data;
    const float* B = (const float*)in[1].data;
    float* O = (float*)out->data;

    // reference
    float* RefC = new float[MM*NN];
    matmul_ref(A, B, RefC);

    float* CurC = new float[MM*NN];
    QFn qs[4] = {q_fp32, q_fp16, q_fp8, q_fp4};
    const char* names[4] = {"fp32","fp16","fp8(E4M3)","fp4(E2M1)"};

    for (int q = 0; q < 4; ++q) {
        run_matmul_q(A, B, CurC, qs[q]);
        float maxa = 0.0f, maxrel = 0.0f;
        float maxabs = 0.0f;
        float refmax = 0.0f;
        for (int i = 0; i < MM*NN; ++i) {
            float d = std::fabs(CurC[i] - RefC[i]);
            if (d > maxabs) maxabs = d;
            float r = std::fabs(RefC[i]);
            if (r > refmax) refmax = r;
        }
        maxrel = maxabs / (refmax + 1e-9f);
        O[q*2+0] = maxabs;
        O[q*2+1] = maxrel;
        std::fprintf(stderr, "[precision] %-10s  max_abs_err=%.3e  max_rel_err=%.3e\n", names[q], maxabs, maxrel);
    }
    delete[] RefC; delete[] CurC;

    out->len = 4*2*sizeof(float);
    return 0;
}
