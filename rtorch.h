// RTorch — universal compute framework: unified formula interface.
// A user formula is any C++ file that implements the two functions below.
// The framework (rtorch.exe) compiles the formula into a DLL, feeds it input
// blobs, allocates the output, times the run, and surface the result.
//
// LICENSE NOTE (LGPL-3.0): this header is a PUBLIC INTERFACE (API type and
// function declarations only) — it contains no implementation of the RTorch
// library. A user formula that implements rtorch_output_size / rtorch_compute
// is the author's own code and is NOT a derivative of the RTorch library, so it
// is not bound by RTorch's LGPL-3.0 license; it remains under the author's
// chosen license. (The framework compiles and loads the formula as a separate
// plugin communicating through this ABI.)
//
// Contract: your formula MUST implement rtorch_output_size and rtorch_compute.
// Everything else (compilation, I/O, device selection, timing) is the
// framework's job. You may also implement the legacy rtorch_main entry.
#ifndef RTORCH_H
#define RTORCH_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// A contiguous byte blob. `data` may be NULL when len == 0.
typedef struct {
    const void* data;
    size_t len;
} rtorch_blob;

// Device selector. 0 = CPU host; >0 = best available accelerator
// (currently the first detected OpenCL GPU). The framework picks it and
// passes it in; the formula may ignore it and run on the host if it likes.
enum { RTORCH_DEVICE_CPU = 0, RTORCH_DEVICE_GPU = 1 };

// Return the number of bytes the output blob needs for the given inputs.
// The framework calls this, allocates that many bytes, then calls compute.
// Return 0 if the formula produces no output.
unsigned long long rtorch_output_size(int n_in, const rtorch_blob* in, int device);

// Perform the computation. Write results into out->data (out->len bytes are
// available). Return 0 on success, non-zero on failure. Use the passed
// `device` if you want accelerator support, else compute on the host.
int rtorch_compute(int n_in, const rtorch_blob* in, rtorch_blob* out, int device);

// ---- Optional GPU compute (device = RTORCH_DEVICE_GPU) ----
// Provide a Vulkan compute kernel as GLSL (#version 450). The framework
// compiles it with glslang, binds the input blobs as SSBO bindings 0..n-1 and
// the output blob as binding n, dispatches, and returns the output. If a
// formula exports this, running under `--device gpu` uses the GPU instead of
// host compute.
const char* rtorch_gpu_kernel(void);

// Optional: workgroup counts (gx, gy, gz). Default gx = ceil(out_elems / 256),
// gy = gz = 1. Export to override.
void rtorch_gpu_groups(int* gx, int* gy, int* gz);

#ifdef __cplusplus
}
#endif

#endif // RTORCH_H
