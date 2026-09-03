// RTorch GPU precision benchmark — 64x64 matmul, operands quantized to
// fp4(E2M1) in-shader (simulated: no native GLSL fp4 type), accumulated fp32.
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
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
layout(local_size_x=256) in;
layout(set=0, binding=0) buffer A { float16_t a[]; };
layout(set=0, binding=1) buffer B { float16_t b[]; };
layout(set=0, binding=2) buffer O { float o[]; };
float q4(float x) {
    if (x == 0.0) return 0.0;
    bool neg = x < 0.0;
    float ax = abs(x);
    float maxf = 6.0;
    if (ax >= maxf) return neg ? -maxf : maxf;
    int e = int(floor(log2(ax)));
    float frac = ax / exp2(float(e)) - 1.0;
    float mi = round(frac * 2.0);
    if (mi >= 2.0) { mi = 0.0; e += 1; }
    float q = exp2(float(e)) * (1.0 + mi / 2.0);
    return neg ? -q : q;
}
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= o.length()) return;
    uint row = i / 64u; uint col = i % 64u;
    float acc = 0.0;
    for (uint k = 0u; k < 64u; k++)
        acc += q4(float(a[row * 64u + k])) * q4(float(b[k * 64u + col]));
    o[i] = acc;
}
)GL";
}
