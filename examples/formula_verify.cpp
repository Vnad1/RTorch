// RTorch universal formula example — implements the unified interface.
// Any user formula = rtorch_output_size + rtorch_compute. This one takes a
// blob of little-endian float32 values and reports statistics (a validation
// over arbitrary data). Contrived on purpose to show "anyone can write a
// formula and have the framework verify / compute it."
#include "rtorch.h"

#include <cstdio>
#include <cstdint>
#include <cmath>

unsigned long long rtorch_output_size(int n_in, const rtorch_blob* in, int device) {
    (void)n_in; (void)in; (void)device;
    return 512; // fixed-size report
}

int rtorch_compute(int n_in, const rtorch_blob* in, rtorch_blob* out, int device) {
    (void)device;
    char* o = (char*)out->data;
    if (n_in < 1 || in->data == nullptr || in->len < 4 || (in->len % 4) != 0) {
        std::snprintf(o, out->len, "[verify] no float32 payload (or unaligned length)\n");
        return 0;
    }
    std::size_t n = in->len / 4;
    const float* f = (const float*)in->data;
    double sum = 0.0, sumsq = 0.0;
    float mn = f[0], mx = f[0];
    for (std::size_t i = 0; i < n; ++i) {
        float v = f[i];
        sum += v;
        sumsq += (double)v * v;
        if (v < mn) mn = v;
        if (v > mx) mx = v;
    }
    double mean = sum / (double)n;
    double var = sumsq / (double)n - mean * mean;
    (void)var;
    int written = std::snprintf(o, out->len,
        "[verify] n=%zu  mean=%.6f  min=%.6f  max=%.6f  rms=%.6f\n",
        n, mean, mn, mx, std::sqrt(sumsq / (double)n));
    out->len = (size_t)written; // report true byte length to the framework
    return 0;
}
