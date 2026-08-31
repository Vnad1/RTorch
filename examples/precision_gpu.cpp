// RTorch GPU precision benchmark — the same 64x64 matmul on the accelerator.
// Two kernels selectable by RTORCH_PREC env at runtime isn't possible in GLSL
// (compile-time), so prec is chosen by which .comp is compiled. This formula
// targets fp16 hardware (float16_t storage + fp16 arithmetic). Operands are
// float16_t in memory; accumulation is fp16 to expose true fp16 rounding.
//
// Matches CPU precision_test.cpp for cross-check (cpu vs gpu fp16 behavior).
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
    // one thread per output element (MM*NN = 4096), 256 per WG.
    *gx = (MM*NN + 255) / 256; *gy = 1; *gz = 1;
}

const char* rtorch_gpu_kernel(void) {
    return R"GL(
#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
layout(local_size_x=256) in;
layout(set=0, binding=0) buffer A { float16_t a[]; };
layout(set=0, binding=1) buffer B { float16_t b[]; };
layout(set=0, binding=2) buffer O { float o[]; };
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= o.length()) return;
    uint row = i / 64u;
    uint col = i % 64u;
    float16_t acc = float16_t(0.0);
    for (uint k = 0u; k < 64u; k++)
        acc += a[row * 64u + k] * b[k * 64u + col];
    o[i] = float(acc);
}
)GL";
}
