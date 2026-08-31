// RTorch GPU formula — implements the unified protocol's GPU path.
// Exports a GLSL compute kernel; the framework compiles it to SPIR-V (glslang),
// runs it on the accelerator (Vulkan via the C++ engine), and returns the
// output blob. Here: elementwise C = a * b (two input float32 arrays).
#include "rtorch.h"

#include <cstddef>

// Output blob size: same number of floats as the first input.
unsigned long long rtorch_output_size(int n_in, const rtorch_blob* in, int device) {
    (void)device;
    if (n_in < 1 || in->data == nullptr) return 0;
    return (unsigned long long)in->len;
}

// Optional: workgroup counts. Provide gx = ceil(out_elems / 256) is the default,
// so we can omit this. The kernel uses local_size_x=256 and guards with
// gl_GlobalInvocationID.
void rtorch_gpu_groups(int* gx, int* gy, int* gz) {
    (void)gx; (void)gy; (void)gz; // use framework default
}

// The GLSL compute kernel. Bindings: 0,1 = inputs, 2 = output.
const char* rtorch_gpu_kernel(void) {
    return R"GL(
#version 450
layout(local_size_x=256) in;
layout(set=0, binding=0) buffer A { float a[]; };
layout(set=0, binding=1) buffer B { float b[]; };
layout(set=0, binding=2) buffer O { float o[]; };
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i < o.length())
        o[i] = a[i] * b[i];
}
)GL";
}
