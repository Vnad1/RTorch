// RTorch formula — elementwise trigonometric functions.
// For each input float x[i], output three values: sin, cos, tan.
//
//   in  : x[]  (n float32)          binding 0
//   out : packed triples            binding 1
//         out[3i+0] = sin(x[i])
//         out[3i+1] = cos(x[i])
//         out[3i+2] = tan(x[i])
//
// The output blob is 3 * n * sizeof(float) bytes. Both CPU (rtorch_compute)
// and GPU (rtorch_gpu_kernel, Vulkan compute) paths are provided, so
// `--device cpu` and `--device gpu` both work. The formula is the author's own
// code and is not a derivative of RTorch (see rtorch.h license note).
#include "rtorch.h"

#include <cstddef>
#include <cstdint>
#include <cmath>

static std::size_t elem_count(int n_in, const rtorch_blob* in) {
    if (n_in < 1 || in->data == nullptr || in->len % 4 != 0) return 0;
    return in->len / 4;
}

// Output: 3 floats per input element.
unsigned long long rtorch_output_size(int n_in, const rtorch_blob* in, int device) {
    (void)device;
    return (unsigned long long)(elem_count(n_in, in) * 3 * 4);
}

// CPU path: compute sin/cos/tan per element on the host.
int rtorch_compute(int n_in, const rtorch_blob* in, rtorch_blob* out, int device) {
    (void)device;
    std::size_t n = elem_count(n_in, in);
    if (n == 0) return 1;
    const float* x = (const float*)in->data;
    std::size_t out_needed = n * 3 * sizeof(float);
    if (out->data == nullptr || out->len < out_needed) return 2;
    float* o = (float*)out->data;
    for (std::size_t i = 0; i < n; ++i) {
        float v = x[i];
        o[3 * i + 0] = std::sin(v);
        o[3 * i + 1] = std::cos(v);
        o[3 * i + 2] = std::tan(v);
    }
    return 0;
}

// GPU path: Vulkan compute kernel, binding 0 = input, binding 1 = output.
const char* rtorch_gpu_kernel(void) {
    return R"GL(
#version 450
layout(local_size_x=256) in;
layout(set=0, binding=0) buffer X { float x[]; };
layout(set=0, binding=1) buffer O { float o[]; };
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i < o.length() / 3u) {
        float v = x[i];
        o[3u * i + 0u] = sin(v);
        o[3u * i + 1u] = cos(v);
        o[3u * i + 2u] = tan(v);
    }
}
)GL";
}

// Default workgroup counts: gx = ceil(out_elems/256) handled by the framework;
// override to align by input count (one thread per element).
void rtorch_gpu_groups(int* gx, int* gy, int* gz) {
    std::size_t n = 0;
    // The framework calls this before dispatch; we don't have the input here, so
    // fall back to the framework default (gx = ceil(out_elems/256)).
    (void)n;
    *gx = 0; // 0 signals "use framework default"
    *gy = 1;
    *gz = 1;
}
