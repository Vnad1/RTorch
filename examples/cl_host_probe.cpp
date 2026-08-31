// Independent minimal OpenCL host test — NOT loaded via RTorch.
// Bounds whether the GPU OpenCL path works at all outside the RTorch DLL context.
// Parallel Leibniz series in float, host-reduced to pi.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <Windows.h>

typedef int (*clGetPlatformIDs_t)(unsigned, void**, unsigned*);
typedef int (*clGetDeviceIDs_t)(void*, unsigned long long, unsigned, void**, unsigned*);
typedef int (*clGetDeviceInfo_t)(void*, unsigned, size_t, void*, size_t*);
typedef void* (*clCreateContext_t)(const int*, unsigned, void*, void*, void*, int*);
typedef void* (*clCreateCommandQueue_t)(void*, void*, unsigned long long, int*);
typedef void* (*clCreateBuffer_t)(void*, unsigned long long, size_t, void*, int*);
typedef void* (*clCreateProgramWithSource_t)(void*, unsigned, const char**, const size_t*, int*);
typedef int (*clBuildProgram_t)(void*, unsigned, void*, const char*, void*, void*);
typedef void* (*clCreateKernel_t)(void*, const char*, int*);
typedef int (*clSetKernelArg_t)(void*, unsigned, size_t, const void*);
typedef int (*clEnqueueNDRangeKernel_t)(void*, unsigned, const size_t*, const size_t*, const size_t*, unsigned, void*, void*);
typedef int (*clEnqueueReadBuffer_t)(void*, void*, int, size_t, size_t, void*, unsigned, void*, void*);
typedef int (*clFinish_t)(void*);
typedef int (*clGetKernelInfo_t)(void*, unsigned, size_t, void*, size_t*);

template <typename T> static T cs(void* h, const char* n) { return (T)GetProcAddress((HMODULE)h, n); }

static const char* KERN = R"CL(
__kernel void pi_kernel(__global float* out) {
    out[get_global_id(0)] = 42.0f;
}
)CL";

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    HMODULE cl = LoadLibraryA("OpenCL.dll");
    if (!cl) { printf("no OpenCL.dll\n"); return 1; }
    auto pGP=cs<clGetPlatformIDs_t>(cl,"clGetPlatformIDs");
    auto pGD=cs<clGetDeviceIDs_t>(cl,"clGetDeviceIDs");
    auto pGI=cs<clGetDeviceInfo_t>(cl,"clGetDeviceInfo");
    auto pCC=cs<clCreateContext_t>(cl,"clCreateContext");
    auto pCQ=cs<clCreateCommandQueue_t>(cl,"clCreateCommandQueue");
    auto pCB=cs<clCreateBuffer_t>(cl,"clCreateBuffer");
    auto pCP=cs<clCreateProgramWithSource_t>(cl,"clCreateProgramWithSource");
    auto pBd=cs<clBuildProgram_t>(cl,"clBuildProgram");
    auto pCK=cs<clCreateKernel_t>(cl,"clCreateKernel");
    auto pSA=cs<clSetKernelArg_t>(cl,"clSetKernelArg");
    auto pNR=cs<clEnqueueNDRangeKernel_t>(cl,"clEnqueueNDRangeKernel");
    auto pRD=cs<clEnqueueReadBuffer_t>(cl,"clEnqueueReadBuffer");
    auto pFn=cs<clFinish_t>(cl,"clFinish");
    auto pKI=cs<clGetKernelInfo_t>(cl,"clGetKernelInfo");
    if(!pGP||!pGD||!pGI||!pCC||!pCQ||!pCB||!pCP||!pBd||!pCK||!pSA||!pNR||!pRD||!pFn){printf("missing sym\n");return 1;}

    unsigned np=0; pGP(0,nullptr,&np);
    printf("platforms=%u\n",np);
    void** plats=(void**)malloc(sizeof(void*)*np);
    pGP(np,plats,&np);
    void* dev=nullptr;
    for(unsigned p=0;p<np&&!dev;++p){
        unsigned nd=0; pGD(plats[p],1,0,nullptr,&nd);
        if(!nd)continue;
        void** ds=(void**)malloc(sizeof(void*)*nd);
        pGD(plats[p],1,nd,ds,&nd);
        dev=ds[0];
        char nm[256]={0}; size_t sz=0; pGI(dev,0x102B,255,nm,&sz);
        printf("device=%s\n",nm);
    }
    if(!dev){printf("no gpu\n");return 1;}
    int err=0;
    void* ctx=pCC(nullptr,1,&dev,nullptr,nullptr,&err); printf("ctx err=%d\n",err);
    void* q=pCQ(ctx,dev,0,&err); printf("queue err=%d\n",err);
    unsigned g=1; size_t bytes=g*sizeof(float);
    void* out=pCB(ctx,1,bytes,nullptr,&err); printf("buf err=%d\n",err);
    const char* s=KERN; void* pr=pCP(ctx,1,&s,nullptr,&err); printf("prog err=%d\n",err);
    err=pBd(pr,1,&dev,nullptr,nullptr,nullptr); printf("build err=%d\n",err);
    void* k=pCK(pr,"pi_kernel",&err); printf("kernel err=%d\n",err);
    unsigned nargs=0; size_t sz=0; pKI(k,0x1191,4,&nargs,&sz); printf("kernel numargs=%u\n",nargs);
    int a0=pSA(k,0,sizeof(void*),&out); printf("setarg0=%d\n",a0);
    size_t gs=g, ls=1, zero=0;
    int ke=pNR(q,1,&zero,&gs,&ls,0,nullptr,nullptr); printf("ndrange=%d\n",ke);
    int fe=pFn(q); printf("finish=%d\n",fe);
    float* h=(float*)malloc(bytes);
    int re=pRD(q,out,1,0,bytes,h,0,nullptr,nullptr); printf("read=%d h0=%f\n",re,h[0]);
    return 0;
}
