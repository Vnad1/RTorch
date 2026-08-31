// RTorch spike_cell v1 — integrate-and-fire state model with SQUARE
// self-limiting (saturation) and channel cross-coupling. Brain-inspired,
// non-transformer.
//
// The square term x - alpha*x^2 gives state a natural upper bound (membrane
// potential cannot grow without limit — like a neuron with a ceiling), and the
// A matrix couples channels (one dimension influences another), so the model
// has real dynamics rather than per-channel isolation.
//
// Input:  [B, T, D] float32 (header [B,T,D] first 3 floats, then data).
// Evolution (leaky integrate + square saturation + fire/reset):
//     z       = A·v[t-1] + drive·u[t]           # coupled linear evolution
//     v[t]    = z - alpha·(z·z)                 # square saturation (self-limit)
//     if v[t] > theta: fire, v[t] = v_reset
// Parameters (v0 fixed, exposed via headers; learned later):
//     A[DxD] : channel coupling,      alpha : square saturation strength
//     leak   : linear decay,          drive : observation injection gain
// Output per-b: final_state(D) | spike_count(1) | peak(D)
#include "rtorch.h"
#include <cstddef>
#include <cstdint>
#include <cmath>

static const float ALPHA   = 0.02f;   // square saturation strength
static const float LEAK    = 0.05f;   // linear decay
static const float DRIVE   = 1.0f;    // observation gain
static const float THETA   = 1000.0f; // high => rarely fires (v0 test)
static const float V_RESET = 0.0f;

unsigned long long rtorch_output_size(int n_in, const rtorch_blob* in, int device) {
    (void)device;
    if (n_in < 1) return 0;
    const float* x = (const float*)in[0].data;
    size_t nf = in[0].len / sizeof(float);
    if (nf < 3) return 0;
    size_t B = (size_t)x[0], T = (size_t)x[1], D = (size_t)x[2];
    size_t per_b = D + 1 + D;
    return (unsigned long long)(B * per_b * sizeof(float));
}

int rtorch_compute(int n_in, const rtorch_blob* in, rtorch_blob* out, int device) {
    (void)device;
    if (n_in < 1) return 1;
    const float* x = (const float*)in[0].data;
    size_t nf = in[0].len / sizeof(float);
    if (nf < 3) return 2;
    size_t B = (size_t)x[0], T = (size_t)x[1], D = (size_t)x[2];
    const float* data = x + 3;

    // A: channel coupling (D x D). v0: identity-dominant + small off-diagonal,
    // so channels loosely interact. We build it from a fixed seed pattern.
    float* A = new float[D * D];
    for (size_t i = 0; i < D; ++i) {
        for (size_t j = 0; j < D; ++j) {
            float a = (i == j) ? 0.9f : 0.05f;   // dominant self, weak cross
            A[i * D + j] = a;
        }
    }

    float* O = (float*)out->data;

    for (size_t b = 0; b < B; ++b) {
        float* v = new float[D];
        float* peak = new float[D];
        for (size_t d = 0; d < D; ++d) { v[d] = 0.0f; peak[d] = 0.0f; }
        unsigned int spike_count = 0;
        const float* seq = data + b * T * D;

        for (size_t t = 0; t < T; ++t) {
            for (size_t d = 0; d < D; ++d) {
                // coupled linear evolution: z_d = sum_j A[d][j]*v[j] + drive*u[d]
                float z = 0.0f;
                for (size_t j = 0; j < D; ++j) z += A[d * D + j] * v[j];
                z += DRIVE * seq[t * D + d];
                // linear decay then square saturation
                z = z - LEAK * z;
                float vn = z - ALPHA * (z * z);
                if (vn > peak[d]) peak[d] = vn;
                if (vn > THETA) { spike_count++; vn = V_RESET; }
                v[d] = vn;
            }
        }

        float* ob = O + b * (D + 1 + D);
        for (size_t d = 0; d < D; ++d) ob[d] = v[d];
        ob[D] = (float)spike_count;
        for (size_t d = 0; d < D; ++d) ob[D + 1 + d] = peak[d];

        delete[] v; delete[] peak;
    }
    delete[] A;

    out->len = (unsigned long long)(B * (D + 1 + D) * sizeof(float));
    return 0;
}
