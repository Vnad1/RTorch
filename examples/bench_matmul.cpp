// RTorch GPU benchmark — C = A * B, 1024x1024x1024.
// 8x8 register block per thread, local(16,16) = 256 threads, each WG computes
// a 128x128 output tile (shared A[128][16] + B[16][128] = 16KB, single buffer).
#include "rtorch.h"
#include <cstddef>

#define M 1024
#define K 1024
#define N 1024

unsigned long long rtorch_output_size(int n_in, const rtorch_blob* in, int device) {
    (void)in; (void)device;
    if (n_in < 2) return 0;
    return (unsigned long long)(M * N * sizeof(float));
}

// 16x16 threads * 8x8 block = 128x128 tile per WG -> 8x8 WGs for 1024x1024.
void rtorch_gpu_groups(int* gx, int* gy, int* gz) {
    *gx = N / 128; *gy = M / 128; *gz = 1;
}

const char* rtorch_gpu_kernel(void) {
    return R"GL(
#version 450
layout(local_size_x=16, local_size_y=16) in;
layout(set=0, binding=0) buffer A { float a[]; };
layout(set=0, binding=1) buffer B { float b[]; };
layout(set=0, binding=2) buffer O { float o[]; };
shared float As[128][17];
shared float Bs[17][128];
void main() {
    uint wgx = gl_WorkGroupID.x;
    uint wgy = gl_WorkGroupID.y;
    uint lx = gl_LocalInvocationID.x;
    uint ly = gl_LocalInvocationID.y;
    uint brow = wgy * 128u + ly * 8u;
    uint bcol = wgx * 128u + lx * 8u;
    float acc[64];
    for (int i=0;i<64;i++) acc[i]=0.0;

    for (uint t = 0u; t < 64u; t++) {
        for (uint i=0;i<8u;i++) As[ly*8u+i][lx] = a[(brow+i)*1024u + t*16u + lx];
        for (uint j=0;j<8u;j++) Bs[ly][lx*8u+j] = b[(t*16u+ly)*1024u + bcol + j];
        barrier();
        for (uint kk = 0u; kk < 16u; kk++) {
            for (uint i=0;i<8u;i++) {
                float a0 = As[ly*8u+i][kk];
                for (uint j=0;j<8u;j++) {
                    float bv = Bs[kk][lx*8u+j];
                    acc[i*8u+j] += a0 * bv;
                }
            }
        }
        barrier();
    }
    for (uint i=0;i<8u;i++)
        for (uint j=0;j<8u;j++)
            o[(brow+i)*1024u + bcol + j] = acc[i*8u+j];
}
)GL";
}
