// RTorch Vulkan compute engine — PERSISTENT + DEVICE-RESIDENT multi-pipeline.
//
// Two models, both exported:
//  (OLD) single-shader session: rtorch_vk_init/dispatch/destroy. Kept for the
//        existing gpu.rs / main.rs smoke path. Setup runs once, dispatch loops.
//  (NEW) device-resident: tensors live on the GPU. One device owns a pool of
//        device buffers + host staging, and many pipelines (one per kernel),
//        each bound to specific input/output buffers. Ops chain device-side
//        with NO host copies in between (the real path to 500M throughput).
//
// NEW C ABI:
//   int  rtorch_vk_dev_init() -> dev ctx (>=0), or <0.
//   int  rtorch_vk_alloc(int dev, size_t size) -> buffer index (>=0) or <0.
//   void rtorch_vk_upload(int dev, int buf, const void* data, size_t len);
//   void rtorch_vk_download(int dev, int buf, void* out, size_t len);
//   int  rtorch_vk_pipe_add(int dev, const void* spv, size_t spv_len,
//                           const int* in_bufs, int num_in, int out_buf,
//                           uint32_t gx,uint32_t gy,uint32_t gz) -> pipe id or <0.
//   int  rtorch_vk_pipe_run(int dev, int pipe);   // dispatch + barrier
//   void rtorch_vk_dev_destroy(int dev);
//
// Build:
//   g++ -std=c++17 -shared -O2 -static-libgcc -static-libstdc++ \
//       -I <SDK>/Include vk_engine.cpp <SDK>/Lib/vulkan-1.lib -o rtorch_vk.dll

#include <vulkan/vulkan.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>
#include <algorithm>

#define RTRY(expr) do { VkResult _r = (expr); if (_r != VK_SUCCESS) { \
    std::fprintf(stderr, "[vk-engine] %s -> VkResult %d\n", #expr, (int)_r); return -1; } } while (0)

static bool want_validate() {
    const char* v = std::getenv("RTORCH_VK_VALIDATE");
    return v && v[0] && v[0] != '0';
}

// Host copy helper: one-off submit of a single buffer copy.
static int one_copy(VkDevice dev, VkQueue queue, VkCommandPool pool,
                    VkBuffer src, VkBuffer dst, VkDeviceSize size,
                    VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage) {
    VkCommandBufferAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = pool; ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount = 1;
    VkCommandBuffer cb; RTRY(vkAllocateCommandBuffers(dev, &ai, &cb));
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    RTRY(vkBeginCommandBuffer(cb, &bi));
    { VkBufferCopy cp{0, 0, size}; vkCmdCopyBuffer(cb, src, dst, 1, &cp); }
    vkCmdPipelineBarrier(cb, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 0, nullptr);
    vkEndCommandBuffer(cb);
    VkSubmitInfo si{}; si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1; si.pCommandBuffers = &cb;
    RTRY(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
    RTRY(vkQueueWaitIdle(queue));
    vkFreeCommandBuffers(dev, pool, 1, &cb);
    return 0;
}

// ---- OLD single-shader session (kept for smoke / gpu.rs) ----
struct Ctx {
    VkInstance inst{}; VkPhysicalDevice pdev{}; VkDevice dev{}; VkQueue queue{};
    uint32_t qfam = 0; uint32_t dmem = 0; uint32_t hmem = 0;
    VkShaderModule module{}; VkDescriptorSetLayout dsl{}; VkPipelineLayout layout{};
    VkPipeline pipeline{}; VkDescriptorPool pool{}; VkDescriptorSet dset{};
    VkCommandPool cpool{}; VkCommandBuffer cbuf{};
    std::vector<VkBuffer> cbufs; std::vector<VkDeviceMemory> cmems;
    std::vector<VkBuffer> stbufs; std::vector<VkDeviceMemory> stmems;
    std::vector<VkDeviceSize> sizes; size_t num_inputs = 0; size_t out_len = 0;
};

static std::vector<Ctx*> g_ctxs;

static int find_memtypes2(VkPhysicalDevice pdev, uint32_t* dmem, uint32_t* hmem) {
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(pdev, &mp);
    uint32_t d = mp.memoryTypeCount, h = mp.memoryTypeCount;
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        uint32_t f = mp.memoryTypes[i].propertyFlags;
        if (d == mp.memoryTypeCount && (f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
            && !(f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) d = i;
        if (h == mp.memoryTypeCount &&
            (f & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
               == (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) h = i;
    }
    if (d == mp.memoryTypeCount || h == mp.memoryTypeCount) { std::fputs("[vk-engine] missing mem types\n", stderr); return -1; }
    *dmem = d; *hmem = h; return 0;
}

// Create instance + device + queue; fill dev-common members.
static int init_device(VkInstance* inst, VkPhysicalDevice* pdev, VkDevice* dev,
                       VkQueue* queue, uint32_t* qfam) {
    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO; app.pApplicationName = "rtorch";
    app.applicationVersion = 1; app.pEngineName = "rtorch"; app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO; ici.pApplicationInfo = &app;
    if (want_validate()) { ici.enabledLayerCount = 1; const char* layer = "VK_LAYER_KHRONOS_validation"; ici.ppEnabledLayerNames = &layer;
        ici.enabledExtensionCount = 1; const char* ext = VK_EXT_DEBUG_UTILS_EXTENSION_NAME; ici.ppEnabledExtensionNames = &ext; }
    if (vkCreateInstance(&ici, nullptr, inst) != VK_SUCCESS) return -1;
    uint32_t nd = 0; vkEnumeratePhysicalDevices(*inst, &nd, nullptr);
    std::vector<VkPhysicalDevice> ds(nd ? nd : 1);
    vkEnumeratePhysicalDevices(*inst, &nd, ds.data());
    for (auto pd : ds) {
        uint32_t nq = 0; vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, nullptr);
        std::vector<VkQueueFamilyProperties> qp(nq ? nq : 1);
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, qp.data());
        for (uint32_t i = 0; i < nq; ++i)
            if (qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { *pdev = pd; *qfam = i; goto found; }
    }
found:;
    if (!*pdev) { std::fputs("[vk-engine] no compute dev\n", stderr); vkDestroyInstance(*inst, nullptr); return -1; }
    float prio = 1.0f;
    VkDeviceQueueCreateInfo dqci{};
    dqci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    dqci.queueFamilyIndex = *qfam; dqci.queueCount = 1; dqci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &dqci;
    if (vkCreateDevice(*pdev, &dci, nullptr, dev) != VK_SUCCESS) {
        vkDestroyInstance(*inst, nullptr); return -1;
    }
    vkGetDeviceQueue(*dev, *qfam, 0, queue);
    return 0;
}

static int mk_buf(VkDevice dev, VkDeviceSize sz, uint32_t usage, uint32_t mt,
                  VkBuffer* b, VkDeviceMemory* m) {
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO; bci.size = sz;
    bci.usage = usage; bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(dev, &bci, nullptr, b) != VK_SUCCESS) return -1;
    VkMemoryRequirements req{}; vkGetBufferMemoryRequirements(dev, *b, &req);
    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = req.size; mai.memoryTypeIndex = mt;
    if (vkAllocateMemory(dev, &mai, nullptr, m) != VK_SUCCESS) return -1;
    if (vkBindBufferMemory(dev, *b, *m, 0) != VK_SUCCESS) return -1;
    return 0;
}

extern "C" {

__declspec(dllexport) int rtorch_vk_init(
    const void* spv, size_t spv_len,
    const size_t* input_sizes, int num_inputs,
    size_t out_len)
{
    if (!spv || spv_len % 4 != 0) { std::fputs("[vk-engine] bad spv\n", stderr); return -1; }
    if (num_inputs < 0) { std::fputs("[vk-engine] bad num_inputs\n", stderr); return -1; }
    const uint32_t nbuffers = (uint32_t)num_inputs + 1;

    Ctx* c = new Ctx();
    c->num_inputs = (size_t)num_inputs;
    c->out_len = out_len;
    if (init_device(&c->inst, &c->pdev, &c->dev, &c->queue, &c->qfam) != 0) { delete c; return -1; }
    if (find_memtypes2(c->pdev, &c->dmem, &c->hmem) != 0) { vkDestroyDevice(c->dev, nullptr); vkDestroyInstance(c->inst, nullptr); delete c; return -1; }

    VkShaderModuleCreateInfo smci{};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = spv_len; smci.pCode = (const uint32_t*)spv;
    RTRY(vkCreateShaderModule(c->dev, &smci, nullptr, &c->module));

    std::vector<VkDescriptorSetLayoutBinding> bindings(nbuffers);
    for (uint32_t i = 0; i < nbuffers; ++i) {
        bindings[i].binding = i; bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1; bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dsci{};
    dsci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dsci.bindingCount = nbuffers; dsci.pBindings = bindings.data();
    if (vkCreateDescriptorSetLayout(c->dev, &dsci, nullptr, &c->dsl) != VK_SUCCESS) return -1;

    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1; plci.pSetLayouts = &c->dsl;
    if (vkCreatePipelineLayout(c->dev, &plci, nullptr, &c->layout) != VK_SUCCESS) return -1;

    VkPipelineShaderStageCreateInfo stage{};
    stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; stage.module = c->module; stage.pName = "main";
    VkComputePipelineCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage = stage; cpci.layout = c->layout;
    if (vkCreateComputePipelines(c->dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &c->pipeline) != VK_SUCCESS) return -1;

    c->sizes.resize(nbuffers);
    c->cbufs.resize(nbuffers); c->cmems.resize(nbuffers);
    c->stbufs.resize(nbuffers); c->stmems.resize(nbuffers);
    for (uint32_t i = 0; i < nbuffers; ++i) {
        VkDeviceSize sz = (i < (uint32_t)num_inputs) ? input_sizes[i] : out_len;
        if (sz == 0) sz = 4;
        c->sizes[i] = sz;
        VkBufferUsageFlags cu = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        if (i < (uint32_t)num_inputs) cu |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        else cu |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        if (mk_buf(c->dev, sz, cu, c->dmem, &c->cbufs[i], &c->cmems[i]) != 0) return -1;
        VkBufferUsageFlags su = (i < (uint32_t)num_inputs)
            ? VK_BUFFER_USAGE_TRANSFER_DST_BIT : VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        if (mk_buf(c->dev, sz, su, c->hmem, &c->stbufs[i], &c->stmems[i]) != 0) return -1;
    }

    VkDescriptorPoolSize psz{}; psz.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; psz.descriptorCount = nbuffers;
    VkDescriptorPoolCreateInfo dpci{};
    dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets = 1; dpci.poolSizeCount = 1; dpci.pPoolSizes = &psz;
    if (vkCreateDescriptorPool(c->dev, &dpci, nullptr, &c->pool) != VK_SUCCESS) return -1;
    VkDescriptorSetAllocateInfo dsai{};
    dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool = c->pool; dsai.descriptorSetCount = 1; dsai.pSetLayouts = &c->dsl;
    if (vkAllocateDescriptorSets(c->dev, &dsai, &c->dset) != VK_SUCCESS) return -1;
    std::vector<VkDescriptorBufferInfo> dbis(nbuffers);
    std::vector<VkWriteDescriptorSet> writes(nbuffers);
    for (uint32_t i = 0; i < nbuffers; ++i) {
        dbis[i].buffer = c->cbufs[i]; dbis[i].offset = 0; dbis[i].range = VK_WHOLE_SIZE;
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = c->dset; writes[i].dstBinding = i; writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[i].pBufferInfo = &dbis[i];
    }
    vkUpdateDescriptorSets(c->dev, nbuffers, writes.data(), 0, nullptr);

    VkCommandPoolCreateInfo cpci2{};
    cpci2.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO; cpci2.queueFamilyIndex = c->qfam;
    if (vkCreateCommandPool(c->dev, &cpci2, nullptr, &c->cpool) != VK_SUCCESS) return -1;
    VkCommandBufferAllocateInfo cbai{};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = c->cpool; cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbai.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(c->dev, &cbai, &c->cbuf) != VK_SUCCESS) return -1;

    g_ctxs.push_back(c);
    return (int)g_ctxs.size() - 1;
}

__declspec(dllexport) int rtorch_vk_dispatch(
    int ctx, const void* const* inputs,
    uint32_t gx, uint32_t gy, uint32_t gz,
    void* out, double* elapsed_ms, int reuse_input)
{
    if (ctx < 0 || (size_t)ctx >= g_ctxs.size() || !g_ctxs[ctx]) { std::fputs("[vk-engine] bad ctx\n", stderr); return -1; }
    Ctx* c = g_ctxs[ctx];
    for (size_t i = 0; i < c->num_inputs; ++i) {
        if (reuse_input) continue;
        VkDeviceSize sz = c->sizes[i];
        void* p; RTRY(vkMapMemory(c->dev, c->stmems[i], 0, sz, 0, &p));
        std::memcpy(p, inputs[i], sz); vkUnmapMemory(c->dev, c->stmems[i]);
    }
    auto submit = [&](VkCommandBuffer cb, bool copy_up, bool copy_dn, bool no_upload) {
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        RTRY(vkBeginCommandBuffer(cb, &bi));
        for (size_t i = 0; i < c->num_inputs; ++i)
            if (copy_up && !no_upload) { VkBufferCopy cp{0, 0, c->sizes[i]}; vkCmdCopyBuffer(cb, c->stbufs[i], c->cbufs[i], 1, &cp); }
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 0, nullptr, 0, nullptr, 0, nullptr);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, c->pipeline);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, c->layout, 0, 1, &c->dset, 0, nullptr);
        vkCmdDispatch(cb, gx, gy, gz);
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, nullptr, 0, nullptr, 0, nullptr);
        if (copy_dn) { VkBufferCopy cp{0, 0, c->out_len}; vkCmdCopyBuffer(cb, c->cbufs[c->num_inputs], c->stbufs[c->num_inputs], 1, &cp); }
        vkEndCommandBuffer(cb);
        VkSubmitInfo si{}; si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO; si.commandBufferCount = 1; si.pCommandBuffers = &cb;
        RTRY(vkQueueSubmit(c->queue, 1, &si, VK_NULL_HANDLE));
        RTRY(vkQueueWaitIdle(c->queue));
    };
    auto t0 = std::chrono::high_resolution_clock::now();
    submit(c->cbuf, true, true, reuse_input != 0);
    auto t1 = std::chrono::high_resolution_clock::now();
    if (elapsed_ms) *elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (std::getenv("RTORCH_VK_TSTAMP")) {
        int iters = 50;
        auto k0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) {
            VkCommandBufferBeginInfo bi{};
            bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            vkBeginCommandBuffer(c->cbuf, &bi);
            vkCmdBindPipeline(c->cbuf, VK_PIPELINE_BIND_POINT_COMPUTE, c->pipeline);
            vkCmdBindDescriptorSets(c->cbuf, VK_PIPELINE_BIND_POINT_COMPUTE, c->layout, 0, 1, &c->dset, 0, nullptr);
            vkCmdDispatch(c->cbuf, gx, gy, gz);
            vkEndCommandBuffer(c->cbuf);
            VkSubmitInfo si{}; si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO; si.commandBufferCount = 1; si.pCommandBuffers = &c->cbuf;
            vkQueueSubmit(c->queue, 1, &si, VK_NULL_HANDLE); vkQueueWaitIdle(c->queue);
        }
        auto k1 = std::chrono::high_resolution_clock::now();
        double km = std::chrono::duration<double, std::milli>(k1 - k0).count() / iters;
        std::fprintf(stderr, "[vk-engine] PURE KERNEL avg=%.3f ms over %d iters\n", km, iters);
    }
    if (c->out_len > 0) {
        void* p; RTRY(vkMapMemory(c->dev, c->stmems[c->num_inputs], 0, c->out_len, 0, &p));
        std::memcpy(out, p, c->out_len); vkUnmapMemory(c->dev, c->stmems[c->num_inputs]);
    }
    return 0;
}

__declspec(dllexport) void rtorch_vk_destroy(int ctx) {
    if (ctx < 0 || (size_t)ctx >= g_ctxs.size() || !g_ctxs[ctx]) return;
    Ctx* c = g_ctxs[ctx];
    size_t nb = c->num_inputs + 1;
    for (size_t i = 0; i < nb; ++i) {
        if (c->cbufs[i]) vkDestroyBuffer(c->dev, c->cbufs[i], nullptr);
        if (c->cmems[i]) vkFreeMemory(c->dev, c->cmems[i], nullptr);
        if (c->stbufs[i]) vkDestroyBuffer(c->dev, c->stbufs[i], nullptr);
        if (c->stmems[i]) vkFreeMemory(c->dev, c->stmems[i], nullptr);
    }
    if (c->cpool) vkDestroyCommandPool(c->dev, c->cpool, nullptr);
    if (c->pool) vkDestroyDescriptorPool(c->dev, c->pool, nullptr);
    if (c->pipeline) vkDestroyPipeline(c->dev, c->pipeline, nullptr);
    if (c->layout) vkDestroyPipelineLayout(c->dev, c->layout, nullptr);
    if (c->dsl) vkDestroyDescriptorSetLayout(c->dev, c->dsl, nullptr);
    if (c->module) vkDestroyShaderModule(c->dev, c->module, nullptr);
    if (c->dev) vkDestroyDevice(c->dev, nullptr);
    if (c->inst) vkDestroyInstance(c->inst, nullptr);
    g_ctxs[ctx] = nullptr;
    delete c;
}

// ==========================================================================
// NEW: device-resident multi-pipeline model
// ==========================================================================
struct DevBuf {
    VkBuffer dev{}; VkDeviceMemory dm{}; VkBuffer stg{}; VkDeviceMemory sm{};
    VkDeviceSize size = 0;
};
struct DevPipe {
    VkShaderModule module{}; VkDescriptorSetLayout dsl{}; VkPipelineLayout layout{};
    VkPipeline pipeline{}; VkDescriptorSet dset{}; VkCommandBuffer cbuf{};
    uint32_t gx = 1, gy = 1, gz = 1;
};
struct DevCtx {
    VkInstance inst{}; VkPhysicalDevice pdev{}; VkDevice dev{}; VkQueue queue{};
    uint32_t qfam = 0, dmem = 0, hmem = 0;
    VkDescriptorPool pool{}; VkCommandPool cpool{};
    std::vector<DevBuf> bufs;
    std::vector<size_t> freebufs;  // buffer indices reusable
    std::vector<DevPipe> pipes;
    // Batched recording (reduce per-op vkQueueWaitIdle): a shared record buffer.
    VkCommandBuffer rec_cbuf{};  // one command buffer reused for a batch of dispatches
    bool recording = false;
};

static std::vector<DevCtx*> g_devs;

static DevCtx* dev_ref(int dev) {
    if (dev < 0 || (size_t)dev >= g_devs.size() || !g_devs[dev]) return nullptr;
    return g_devs[dev];
}

__declspec(dllexport) int rtorch_vk_dev_init() {
    DevCtx* d = new DevCtx();
    if (init_device(&d->inst, &d->pdev, &d->dev, &d->queue, &d->qfam) != 0) { delete d; return -1; }
    if (find_memtypes2(d->pdev, &d->dmem, &d->hmem) != 0) { vkDestroyDevice(d->dev, nullptr); vkDestroyInstance(d->inst, nullptr); delete d; return -1; }
    VkDescriptorPoolSize psz{}; psz.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; psz.descriptorCount = 4096;
    VkDescriptorPoolCreateInfo dpci{};
    dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets = 1024; dpci.poolSizeCount = 1; dpci.pPoolSizes = &psz;
    if (vkCreateDescriptorPool(d->dev, &dpci, nullptr, &d->pool) != VK_SUCCESS) return -1;
    VkCommandPoolCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO; cpci.queueFamilyIndex = d->qfam;
    if (vkCreateCommandPool(d->dev, &cpci, nullptr, &d->cpool) != VK_SUCCESS) return -1;
    VkCommandBufferAllocateInfo rbai{};
    rbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    rbai.commandPool = d->cpool; rbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; rbai.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(d->dev, &rbai, &d->rec_cbuf) != VK_SUCCESS) return -1;
    g_devs.push_back(d);
    return (int)g_devs.size() - 1;
}

__declspec(dllexport) int rtorch_vk_alloc(int dev, size_t size) {
    DevCtx* d = dev_ref(dev); if (!d) return -1;
    if (size == 0) size = 4;
    for (size_t i = 0; i < d->freebufs.size(); ++i) {
        size_t bi = d->freebufs[i];
        if (d->bufs[bi].size >= size) {
            d->freebufs.erase(d->freebufs.begin() + i);
            return (int)bi;
        }
    }
    DevBuf b{}; b.size = size;
    VkBufferUsageFlags u = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    if (mk_buf(d->dev, size, u, d->dmem, &b.dev, &b.dm) != 0) return -1;
    VkBufferUsageFlags su = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    if (mk_buf(d->dev, size, su, d->hmem, &b.stg, &b.sm) != 0) return -1;
    d->bufs.push_back(b);
    return (int)d->bufs.size() - 1;
}

__declspec(dllexport) void rtorch_vk_upload(int dev, int buf, const void* data, size_t len) {
    DevCtx* d = dev_ref(dev); if (!d || buf < 0 || (size_t)buf >= d->bufs.size()) return;
    DevBuf& b = d->bufs[buf]; size_t n = len < b.size ? len : (size_t)b.size;
    void* p; if (vkMapMemory(d->dev, b.sm, 0, n, 0, &p) != VK_SUCCESS) return;
    std::memcpy(p, data, n); vkUnmapMemory(d->dev, b.sm);
    one_copy(d->dev, d->queue, d->cpool, b.stg, b.dev, n,
             VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
}

__declspec(dllexport) void rtorch_vk_download(int dev, int buf, void* out, size_t len) {
    DevCtx* d = dev_ref(dev); if (!d || buf < 0 || (size_t)buf >= d->bufs.size()) return;
    DevBuf& b = d->bufs[buf]; size_t n = len < b.size ? len : (size_t)b.size;
    one_copy(d->dev, d->queue, d->cpool, b.dev, b.stg, n,
             VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    void* p; if (vkMapMemory(d->dev, b.sm, 0, n, 0, &p) != VK_SUCCESS) return;
    std::memcpy(out, p, n); vkUnmapMemory(d->dev, b.sm);
}

__declspec(dllexport) int rtorch_vk_pipe_add(
    int dev, const void* spv, size_t spv_len,
    const int* in_bufs, int num_in, int out_buf,
    uint32_t gx, uint32_t gy, uint32_t gz)
{
    DevCtx* d = dev_ref(dev); if (!d) return -1;
    if (!spv || spv_len % 4 != 0) { std::fputs("[vk-engine] bad spv\n", stderr); return -1; }
    if (num_in < 0 || out_buf < 0 || (size_t)out_buf >= d->bufs.size()) return -1;
    const uint32_t nb = (uint32_t)num_in + 1;

    DevPipe p{};
    p.gx = gx; p.gy = gy; p.gz = gz;
    VkShaderModuleCreateInfo smci{};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = spv_len; smci.pCode = (const uint32_t*)spv;
    if (vkCreateShaderModule(d->dev, &smci, nullptr, &p.module) != VK_SUCCESS) return -1;

    std::vector<VkDescriptorSetLayoutBinding> bindings(nb);
    for (uint32_t i = 0; i < nb; ++i) {
        bindings[i].binding = i; bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1; bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dsci{};
    dsci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dsci.bindingCount = nb; dsci.pBindings = bindings.data();
    if (vkCreateDescriptorSetLayout(d->dev, &dsci, nullptr, &p.dsl) != VK_SUCCESS) return -1;
    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1; plci.pSetLayouts = &p.dsl;
    if (vkCreatePipelineLayout(d->dev, &plci, nullptr, &p.layout) != VK_SUCCESS) return -1;
    VkPipelineShaderStageCreateInfo stage{};
    stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; stage.module = p.module; stage.pName = "main";
    VkComputePipelineCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage = stage; cpci.layout = p.layout;
    if (vkCreateComputePipelines(d->dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &p.pipeline) != VK_SUCCESS) return -1;

    VkDescriptorSetAllocateInfo dsai{};
    dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool = d->pool; dsai.descriptorSetCount = 1; dsai.pSetLayouts = &p.dsl;
    if (vkAllocateDescriptorSets(d->dev, &dsai, &p.dset) != VK_SUCCESS) return -1;
    std::vector<VkDescriptorBufferInfo> dbis(nb);
    std::vector<VkWriteDescriptorSet> writes(nb);
    for (uint32_t i = 0; i < nb; ++i) {
        int bi = (i < (uint32_t)num_in) ? in_bufs[i] : out_buf;
        if (bi < 0 || (size_t)bi >= d->bufs.size()) return -1;
        dbis[i].buffer = d->bufs[bi].dev; dbis[i].offset = 0; dbis[i].range = VK_WHOLE_SIZE;
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = p.dset; writes[i].dstBinding = i; writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[i].pBufferInfo = &dbis[i];
    }
    vkUpdateDescriptorSets(d->dev, nb, writes.data(), 0, nullptr);

    VkCommandBufferAllocateInfo cbai{};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = d->cpool; cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbai.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(d->dev, &cbai, &p.cbuf) != VK_SUCCESS) return -1;

    d->pipes.push_back(p);
    return (int)d->pipes.size() - 1;
}

__declspec(dllexport) void rtorch_vk_free(int dev, int buf) {
    DevCtx* d = dev_ref(dev); if (!d) return;
    if (buf < 0 || (size_t)buf >= d->bufs.size()) return;
    d->freebufs.push_back((size_t)buf);
}

__declspec(dllexport) int rtorch_vk_pipe_bind(
    int dev, int pipe, const int* in_bufs, int num_in, int out_buf)
{
    DevCtx* d = dev_ref(dev); if (!d) return -1;
    if (pipe < 0 || (size_t)pipe >= d->pipes.size()) return -1;
    DevPipe& p = d->pipes[pipe];
    if (num_in < 0 || out_buf < 0 || (size_t)out_buf >= d->bufs.size()) return -1;
    const uint32_t nb = (uint32_t)num_in + 1;
    std::vector<VkDescriptorBufferInfo> dbis(nb);
    std::vector<VkWriteDescriptorSet> writes(nb);
    for (uint32_t i = 0; i < nb; ++i) {
        int bi = (i < (uint32_t)num_in) ? in_bufs[i] : out_buf;
        if (bi < 0 || (size_t)bi >= d->bufs.size()) return -1;
        dbis[i].buffer = d->bufs[bi].dev; dbis[i].offset = 0; dbis[i].range = VK_WHOLE_SIZE;
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = p.dset; writes[i].dstBinding = i; writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[i].pBufferInfo = &dbis[i];
    }
    vkUpdateDescriptorSets(d->dev, nb, writes.data(), 0, nullptr);
    return 0;
}

__declspec(dllexport) int rtorch_vk_pipe_run(int dev, int pipe, uint32_t gx, uint32_t gy, uint32_t gz) {
    DevCtx* d = dev_ref(dev); if (!d) return -1;
    if (pipe < 0 || (size_t)pipe >= d->pipes.size()) return -1;
    DevPipe& p = d->pipes[pipe];
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    RTRY(vkBeginCommandBuffer(p.cbuf, &bi));
    vkCmdPipelineBarrier(p.cbuf, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 0, nullptr, 0, nullptr, 0, nullptr);
    vkCmdBindPipeline(p.cbuf, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipeline);
    vkCmdBindDescriptorSets(p.cbuf, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout, 0, 1, &p.dset, 0, nullptr);
    vkCmdDispatch(p.cbuf, gx, gy, gz);
    vkCmdPipelineBarrier(p.cbuf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, nullptr, 0, nullptr, 0, nullptr);
    vkEndCommandBuffer(p.cbuf);
    VkSubmitInfo si{}; si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1; si.pCommandBuffers = &p.cbuf;
    RTRY(vkQueueSubmit(d->queue, 1, &si, VK_NULL_HANDLE));
    RTRY(vkQueueWaitIdle(d->queue));
    return 0;
}

// Begin a batched-recording pass on the shared record command buffer.
__declspec(dllexport) int rtorch_vk_dev_begin(int dev) {
    DevCtx* d = dev_ref(dev); if (!d) return -1;
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    RTRY(vkBeginCommandBuffer(d->rec_cbuf, &bi));
    d->recording = true;
    return 0;
}

// Record one dispatch (bind pipeline + descriptor set + dispatch) into the
// record buffer WITHOUT submitting. `pipe` must have been pipe_bind'ed (its
// descriptor set already points at current in/out buffers). No wait per op.
// A COMPUTE->COMPUTE memory barrier is inserted before the dispatch so the
// prior op's writes are visible to this one (data dependency between ops).
__declspec(dllexport) int rtorch_vk_pipe_record(int dev, int pipe, uint32_t gx, uint32_t gy, uint32_t gz) {
    DevCtx* d = dev_ref(dev); if (!d) return -1;
    if (pipe < 0 || (size_t)pipe >= d->pipes.size()) return -1;
    DevPipe& p = d->pipes[pipe];
    VkMemoryBarrier mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(d->rec_cbuf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &mb, 0, nullptr, 0, nullptr);
    vkCmdBindPipeline(d->rec_cbuf, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipeline);
    vkCmdBindDescriptorSets(d->rec_cbuf, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout, 0, 1, &p.dset, 0, nullptr);
    vkCmdDispatch(d->rec_cbuf, gx, gy, gz);
    return 0;
}

// Submit the recorded batch. `wait`=1 drains the queue (final result needed now);
// `wait`=0 leaves it in flight (only for throughput loops that read back later).
__declspec(dllexport) int rtorch_vk_dev_submit(int dev, int wait) {
    DevCtx* d = dev_ref(dev); if (!d) return -1;
    if (!d->recording) return -1;
    vkEndCommandBuffer(d->rec_cbuf);
    d->recording = false;
    VkSubmitInfo si{}; si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1; si.pCommandBuffers = &d->rec_cbuf;
    RTRY(vkQueueSubmit(d->queue, 1, &si, VK_NULL_HANDLE));
    if (wait) RTRY(vkQueueWaitIdle(d->queue));
    return 0;
}

__declspec(dllexport) void rtorch_vk_dev_destroy(int dev) {
    DevCtx* d = dev_ref(dev); if (!d) return;
    if (d->rec_cbuf) vkFreeCommandBuffers(d->dev, d->cpool, 1, &d->rec_cbuf);
    for (auto& p : d->pipes) {
        if (p.cbuf) vkFreeCommandBuffers(d->dev, d->cpool, 1, &p.cbuf);
        if (p.pipeline) vkDestroyPipeline(d->dev, p.pipeline, nullptr);
        if (p.layout) vkDestroyPipelineLayout(d->dev, p.layout, nullptr);
        if (p.dsl) vkDestroyDescriptorSetLayout(d->dev, p.dsl, nullptr);
        if (p.module) vkDestroyShaderModule(d->dev, p.module, nullptr);
    }
    for (auto& b : d->bufs) {
        if (b.dev) vkDestroyBuffer(d->dev, b.dev, nullptr);
        if (b.dm) vkFreeMemory(d->dev, b.dm, nullptr);
        if (b.stg) vkDestroyBuffer(d->dev, b.stg, nullptr);
        if (b.sm) vkFreeMemory(d->dev, b.sm, nullptr);
    }
    if (d->cpool) vkDestroyCommandPool(d->dev, d->cpool, nullptr);
    if (d->pool) vkDestroyDescriptorPool(d->dev, d->pool, nullptr);
    if (d->dev) vkDestroyDevice(d->dev, nullptr);
    if (d->inst) vkDestroyInstance(d->inst, nullptr);
    g_devs[dev] = nullptr;
    delete d;
}

} // extern "C"
