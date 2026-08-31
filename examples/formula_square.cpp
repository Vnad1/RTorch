// RTorch GPU formula: elementwise square, c[i] = x[i] * x[i].
// Uses the unified protocol's GPU path (GLSL kernel compiled by the framework,
// run on the accelerator via the C++ Vulkan engine).
#include "rtorch.h"

#include <cstddef>

// Output blob: same number of floats as the single input.
unsigned long long rtorch_output_size(int n_in, const rtorch_blob* in, int device) {
    (void)device;
    if (n_in < 1 || in->data == nullptr || in->len % 4 != 0) return 0;
    return (unsigned long long)in->len;
}

// GLSL compute kernel: binding 0 = input x[], binding 1 = output o[].
const char* rtorch_gpu_kernel(void) {
    return R"GL(
#version 450
layout(local_size_x=256) in;
layout(set=0, binding=0) buffer X { float x[]; };
layout(set=0, binding=1) buffer O { float o[]; };
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i < o.length())
        o[i] = x[i] * x[i];
}
)GL";
}
