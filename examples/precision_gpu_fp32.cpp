// RTorch GPU precision benchmark — 64x64 matmul in native fp32 on the GPU.
#include "rtorch.h"
#include <cstddef>

#define MM 64
#define NN 64
#define KK 64

unsigned long long rtorch_output_size(int n_in, const rtorch_blob* in, int device) {
    (void)in; (void)device;
    if (n_in < 2) return 0;
    return (unsigned long long)(MM * NN * sizeof(float));
}

void rtorch_gpu_groups(int* gx, int* gy, int* gz) {
    *gx = (MM*NN + 255) / 256; *gy = 1; *gz = 1;
}

const char* rtorch_gpu_kernel(void) {
    return R"GL(
#version 450
layout(local_size_x=256) in;
layout(set=0, binding=0) buffer A { float a[]; };
layout(set=0, binding=1) buffer B { float b[]; };
layout(set=0, binding=2) buffer O { float o[]; };
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= o.length()) return;
    uint row = i / 64u; uint col = i % 64u;
    float acc = 0.0;
    for (uint k = 0u; k < 64u; k++)
        acc += a[row * 64u + k] * b[k * 64u + col];
    o[i] = acc;
}
)GL";
}
