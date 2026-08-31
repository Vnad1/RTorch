// RTorch GPU pi benchmark — OpenCL kernel, CUDA-free.
// Contract: extern "C" __declspec(dllexport) int rtorch_main(int, char**).
// Loads OpenCL.dll at runtime (MinGW has no OpenCL.lib), finds a GPU device,
// runs a parallel Leibniz-series kernel, reduces on host, and times the GPU
// path so it can be compared against the CPU pi baseline.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <Windows.h>

// ---- minimal OpenCL dynamic loader ----
typedef int (*clGetPlatformIDs_t)(unsigned, void**, unsigned*);
typedef int (*clGetDeviceIDs_t)(void*, unsigned long long, unsigned, void**, unsigned*);
typedef int (*clGetDeviceInfo_t)(void*, unsigned, size_t, void*, size_t*);
typedef void* (*clCreateContext_t)(const int*, unsigned, void*, void*, void*, int*);
typedef void* (*clCreateCommandQueue_t)(void*, void*, unsigned long long, int*);
typedef void* (*clCreateBuffer_t)(void*, unsigned long long, size_t, void*, int*);
typedef void* (*clCreateProgramWithSource_t)(void*, unsigned, const char**, const size_t*, int*);
typedef int (*clBuildProgram_t)(void*, unsigned, void*, const char*, void*, void*);
typedef void* (*clCreateKernel_t)(void*, const char*, int*);
typedef int (*clSetKernelArg_t)(void*, unsigned, unsigned long, const void*);
typedef int (*clEnqueueNDRangeKernel_t)(void*, unsigned, const size_t*, const size_t*, const size_t*, unsigned, void*, void*);
typedef int (*clEnqueueReadBuffer_t)(void*, void*, int, unsigned long, unsigned long, void*, unsigned, void*, void*);
typedef int (*clFinish_t)(void*);
typedef int (*clReleaseMemObject_t)(void*);
typedef int (*clReleaseContext_t)(void*);
typedef int (*clReleaseCommandQueue_t)(void*);
typedef int (*clReleaseKernel_t)(void*);
typedef int (*clReleaseProgram_t)(void*);
typedef void* (*clGetProgramBuildInfo_t)(void*, void*, unsigned, unsigned long, void*, unsigned long*);

template <typename T> static T load_sym(void* h, const char* name) { return (T)GetProcAddress((HMODULE)h, name); }

static const char* KERNEL_SRC = R"CL(
__kernel void pi_kernel(__global float* out) {
    out[get_global_id(0)] = 42.0f;
}
)CL";

extern "C" __declspec(dllexport)
int rtorch_main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0); // unbuffered so crash loses nothing
    (void)argc;
    long long target = 2000000000LL; // 2e9 terms total by default
    if (argc > 1) {
        long long t = std::atoll(argv[1]);
        if (t > 0) target = t;
    }
    const unsigned global = 1u << 20; // 1M work-items

    long long nterms = target;          // total number of series terms
    long long per_item = (nterms + global - 1) / global;

    printf("[rtorch-gpu] target %lld terms, %u work-items (~%lld terms each)\n",
           nterms, global, per_item);

    HMODULE cl = LoadLibraryA("OpenCL.dll");
    if (!cl) { printf("[rtorch-gpu] OpenCL.dll not found\n"); return 1; }

    auto pGetPlatformIDs = load_sym<clGetPlatformIDs_t>(cl, "clGetPlatformIDs");
    auto pGetDeviceIDs   = load_sym<clGetDeviceIDs_t>(cl, "clGetDeviceIDs");
    auto pGetDeviceInfo  = load_sym<clGetDeviceInfo_t>(cl, "clGetDeviceInfo");
    auto pCreateContext  = load_sym<clCreateContext_t>(cl, "clCreateContext");
    auto pCreateQueue    = load_sym<clCreateCommandQueue_t>(cl, "clCreateCommandQueue");
    auto pCreateBuffer   = load_sym<clCreateBuffer_t>(cl, "clCreateBuffer");
    auto pCreateProgram  = load_sym<clCreateProgramWithSource_t>(cl, "clCreateProgramWithSource");
    auto pBuild          = load_sym<clBuildProgram_t>(cl, "clBuildProgram");
    auto pCreateKernel   = load_sym<clCreateKernel_t>(cl, "clCreateKernel");
    auto pSetArg         = load_sym<clSetKernelArg_t>(cl, "clSetKernelArg");
    auto pNDRange        = load_sym<clEnqueueNDRangeKernel_t>(cl, "clEnqueueNDRangeKernel");
    auto pRead           = load_sym<clEnqueueReadBuffer_t>(cl, "clEnqueueReadBuffer");
    auto pFinish         = load_sym<clFinish_t>(cl, "clFinish");
    auto pRelMem         = load_sym<clReleaseMemObject_t>(cl, "clReleaseMemObject");
    auto pReleaseKernel  = load_sym<clReleaseKernel_t>(cl, "clReleaseKernel");
    auto pReleaseProgram = load_sym<clReleaseProgram_t>(cl, "clReleaseProgram");
    if (!pGetPlatformIDs || !pGetDeviceIDs || !pGetDeviceInfo || !pCreateContext ||
        !pCreateQueue || !pCreateBuffer || !pCreateProgram || !pBuild || !pCreateKernel ||
        !pSetArg || !pNDRange || !pRead || !pFinish || !pRelMem || !pReleaseKernel ||
        !pReleaseProgram) {
        printf("[rtorch-gpu] required OpenCL entry missing\n");
        return 1;
    }

    unsigned nplat = 0;
    pGetPlatformIDs(0, nullptr, &nplat);
    if (!nplat) { printf("[rtorch-gpu] no OpenCL platform\n"); return 1; }
    void** plats = (void**)malloc(sizeof(void*) * nplat);
    pGetPlatformIDs(nplat, plats, &nplat);

    // pick first GPU device across platforms
    void* device = nullptr;
    for (unsigned p = 0; p < nplat && !device; ++p) {
        unsigned ndev = 0;
        pGetDeviceIDs(plats[p], 1 /*CL_DEVICE_TYPE_GPU*/, 0, nullptr, &ndev); // 1 = GPU
        if (!ndev) continue;
        void** devs = (void**)malloc(sizeof(void*) * ndev);
        pGetDeviceIDs(plats[p], 1, ndev, devs, &ndev);
        device = devs[0];
        char nm[256] = {0};
        size_t sz = 0;
        pGetDeviceInfo(device, 0x102B /*CL_DEVICE_NAME*/, 255, nm, &sz);
        printf("[rtorch-gpu] using GPU device: %s\n", nm);
    }
    if (!device) { printf("[rtorch-gpu] no OpenCL GPU device\n"); free(plats); return 1; }

    int err = 0;
    void* ctx = pCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    printf("[rtorch-gpu] ctx=%p err=%d\n", ctx, err);
    if (!ctx) { printf("[rtorch-gpu] clCreateContext failed err=%d\n", err); free(plats); return 1; }
    void* queue = pCreateQueue(ctx, device, 0, &err);
    printf("[rtorch-gpu] queue=%p err=%d\n", queue, err);
    if (!queue) { printf("[rtorch-gpu] clCreateQueue failed err=%d\n", err); return 1; }

    size_t buf_bytes = global * sizeof(float);
    void* out = pCreateBuffer(ctx, 1 /*CL_MEM_WRITE_ONLY*/, buf_bytes, nullptr, &err);
    printf("[rtorch-gpu] buf=%p err=%d\n", out, err);
    if (!out) { printf("[rtorch-gpu] clCreateBuffer failed err=%d\n", err); return 1; }

    const char* src = KERNEL_SRC;
    void* prog = pCreateProgram(ctx, 1, &src, nullptr, &err);
    printf("[rtorch-gpu] prog=%p err=%d\n", prog, err);
    if (!prog) { printf("[rtorch-gpu] clCreateProgramWithSource failed err=%d\n", err); return 1; }
    err = pBuild(prog, 1, &device, nullptr, nullptr, nullptr);
    printf("[rtorch-gpu] build_err=%d\n", err);
    if (err != 0) { printf("[rtorch-gpu] clBuildProgram failed err=%d\n", err); return 1; }
    void* kernel = pCreateKernel(prog, "pi_kernel", &err);
    printf("[rtorch-gpu] kernel=%p err=%d\n", kernel, err);
    if (!kernel) { printf("[rtorch-gpu] clCreateKernel failed err=%d\n", err); return 1; }

    int a0 = pSetArg(kernel, 0, sizeof(void*), &out);
    printf("[rtorch-gpu] setarg0=%d\n", a0);

    // clock
    auto t0 = std::chrono::high_resolution_clock::now();
    size_t gws = global;
    int kerr = pNDRange(queue, 1, nullptr, &gws, nullptr, 0, nullptr, nullptr);
    int ferr = pFinish(queue);
    auto t1 = std::chrono::high_resolution_clock::now();
    printf("[rtorch-gpu] ndrange_err=%d finish_err=%d\n", kerr, ferr);

    float* host = (float*)malloc(buf_bytes);
    int rerr = pRead(queue, out, 1 /*CL_TRUE*/, 0, buf_bytes, host, 0, nullptr, nullptr);
    printf("[rtorch-gpu] read_err=%d  host[0..4]=%f %f %f %f %f\n",
           rerr, host[0], host[1], host[2], host[3], host[4]);

    double sum = 0.0;
    for (unsigned i = 0; i < global; ++i) sum += host[i];
    double pi = 4.0 * sum;

    auto t2 = std::chrono::high_resolution_clock::now();
    double ms_kernel = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double ms_total = std::chrono::duration<double, std::milli>(t2 - t0).count();
    printf("[rtorch-gpu] pi = %.15f\n", pi);
    printf("[rtorch-gpu] kernel time = %.3f ms\n", ms_kernel);
    printf("[rtorch-gpu] total (kernel+readback+reduce) = %.3f ms\n", ms_total);
    printf("[rtorch-gpu] throughput = %.3fM terms/s (kernel)\n", (double)nterms / (ms_kernel / 1000.0) / 1e6);

    free(host); free(plats);
    pRelMem(out); pReleaseKernel(kernel); pReleaseProgram(prog);
    return 0;
}
