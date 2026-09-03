// Standalone driver to run the Vulkan engine under MSVC ASan/UBSan (host-side
// memory/UB in vk_engine.cpp). Compiles vk_engine.cpp + this main into one
// ASan-instrumented exe; loads examples/vec_mul.spv and does init/dispatch/destroy.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>

extern "C" int rtorch_vk_init(const void* spv, size_t spv_len, const size_t* input_sizes, int num_inputs, size_t out_len);
extern "C" int rtorch_vk_dispatch(int ctx, const void* const* inputs, uint32_t gx, uint32_t gy, uint32_t gz, void* out, double* elapsed_ms, int reuse_input);
extern "C" void rtorch_vk_destroy(int ctx);

int main() {
    FILE* f = fopen("examples/vec_mul.spv", "rb");
    if (!f) { printf("no spv\n"); return 2; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> spv(sz);
    if (fread(spv.data(), 1, sz, f) != (size_t)sz) { printf("read fail\n"); return 2; }
    fclose(f);

    const size_t n = 8;
    const size_t in_sizes[2] = { n * 4, n * 4 };
    int ctx = rtorch_vk_init(spv.data(), spv.size(), in_sizes, 2, n * 4);
    if (ctx < 0) { printf("init fail ctx=%d\n", ctx); return 1; }

    float a[8], b[8];
    for (int i = 0; i < 8; i++) { a[i] = (float)i; b[i] = (float)i * 2.0f; }
    const void* inputs[2] = { a, b };
    float out[8];
    double ms = 0.0;
    int rc = rtorch_vk_dispatch(ctx, inputs, 1, 1, 1, out, &ms, 0);
    rtorch_vk_destroy(ctx);
    printf("rc=%d out[0]=%f out[3]=%f ms=%.3f\n", rc, out[0], out[3], ms);
    return rc == 0 ? 0 : 1;
}
