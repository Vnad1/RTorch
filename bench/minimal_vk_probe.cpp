// Minimal standalone Vulkan compute reproducer using the OFFICIAL header +
// vulkan-1.lib (in contrast to the hand-rolled Rust FFI). Mirrors the mono
// path: 1 storage-buffer SSBO, single-group dispatch, read back. Reads real
// SPIR-V from mono.spv (compile examples/mono.comp with glslangValidator).
// If this CRASHES too -> environment/driver issue; if it runs clean -> the
// Rust FFI has a remaining bug.
//
// Build (MinGW against SDK import lib):
//   g++ -std=c++17 -I <SDK>/Include minimal_vk_probe.cpp <SDK>/Lib/vulkan-1.lib -o minimal_vk_probe.exe

#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <fstream>
#include <chrono>

#define CHECK(expr) do { VkResult _r = (expr); if (_r != VK_SUCCESS) { \
    printf("FAIL at line %d: %s -> VkResult %d\n", __LINE__, #expr, (int)_r); return 1; } } while (0)

int main() {
    // Load compiled SPIR-V.
    std::ifstream f("examples/mono.spv", std::ios::binary);
    if (!f) { printf("cannot open examples/mono.spv (run glslang first)\n"); return 1; }
    std::vector<char> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (bytes.size() % 4 != 0) { printf("spv not 4-aligned\n"); return 1; }
    printf("loaded spv %zu bytes\n", bytes.size());

    const char* enable_validation = std::getenv("RTORCH_VK_VALIDATE");
    bool use_layer = enable_validation && enable_validation[0] != '\0' && enable_validation[0] != '0';

    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "rtorch_probe";
    app.applicationVersion = 1;
    app.pEngineName = "none";
    app.apiVersion = VK_API_VERSION_1_1;

    const char* layer = "VK_LAYER_KHRONOS_validation";
    const char* ext = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;

    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;
    if (use_layer) { ici.enabledLayerCount = 1; ici.ppEnabledLayerNames = &layer; }
    if (use_layer) { ici.enabledExtensionCount = 1; ici.ppEnabledExtensionNames = &ext; }
    VkInstance inst;
    CHECK(vkCreateInstance(&ici, nullptr, &inst));

    VkPhysicalDevice pdev = VK_NULL_HANDLE;
    uint32_t chosen_family = 0;
    {
        uint32_t ndev = 0;
        CHECK(vkEnumeratePhysicalDevices(inst, &ndev, nullptr));
        std::vector<VkPhysicalDevice> devs(ndev);
        CHECK(vkEnumeratePhysicalDevices(inst, &ndev, devs.data()));
        for (VkPhysicalDevice pd : devs) {
            uint32_t nqf = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, nullptr);
            std::vector<VkQueueFamilyProperties> qfp(nqf);
            vkGetPhysicalDeviceQueueFamilyProperties(pd, &nqf, qfp.data());
            for (uint32_t i = 0; i < nqf; ++i) {
                if (qfp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { pdev = pd; chosen_family = i; break; }
            }
            if (pdev != VK_NULL_HANDLE) break;
        }
    }
    if (pdev == VK_NULL_HANDLE) { printf("no compute device\n"); return 1; }
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(pdev, &props);
    printf("using device: %s\n", props.deviceName);

    float prio = 1.0f;
    VkDeviceQueueCreateInfo dqci{};
    dqci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    dqci.queueFamilyIndex = chosen_family; dqci.queueCount = 1; dqci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &dqci;
    VkDevice dev;
    CHECK(vkCreateDevice(pdev, &dci, nullptr, &dev));
    VkQueue queue;
    vkGetDeviceQueue(dev, chosen_family, 0, &queue);
    printf("device + queue created\n");

    VkShaderModuleCreateInfo smci{};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = bytes.size();
    smci.pCode = (const uint32_t*)bytes.data();
    VkShaderModule module;
    CHECK(vkCreateShaderModule(dev, &smci, nullptr, &module));
    printf("shader module created\n");

    // descriptor set layout: 1 storage buffer
    VkDescriptorSetLayoutBinding bind{};
    bind.binding = 0; bind.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bind.descriptorCount = 1; bind.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo dsci{};
    dsci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dsci.bindingCount = 1; dsci.pBindings = &bind;
    VkDescriptorSetLayout dsl;
    CHECK(vkCreateDescriptorSetLayout(dev, &dsci, nullptr, &dsl));

    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1; plci.pSetLayouts = &dsl;
    VkPipelineLayout layout;
    CHECK(vkCreatePipelineLayout(dev, &plci, nullptr, &layout));

    VkPipelineShaderStageCreateInfo stage{};
    stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = module;
    stage.pName = "main";
    VkComputePipelineCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage = stage; cpci.layout = layout;
    VkPipeline pipeline;
    CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipeline));
    printf("compute pipeline created\n");

    // buffer (host-visible coherent)
    const uint32_t N = 8;
    const VkDeviceSize buf_size = N * sizeof(float);
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = buf_size; bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    VkShaderStageFlags ssf = VK_SHADER_STAGE_COMPUTE_BIT; (void)ssf;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer buffer;
    CHECK(vkCreateBuffer(dev, &bci, nullptr, &buffer));
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(dev, buffer, &req);

    VkPhysicalDeviceMemoryProperties memp{};
    vkGetPhysicalDeviceMemoryProperties(pdev, &memp);
    uint32_t mt = 0;
    for (; mt < memp.memoryTypeCount; ++mt) {
        if ((memp.memoryTypes[mt].propertyFlags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
            (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) break;
    }
    if (mt == memp.memoryTypeCount) { printf("no host-visible mem\n"); return 1; }
    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = req.size; mai.memoryTypeIndex = mt;
    VkDeviceMemory mem;
    CHECK(vkAllocateMemory(dev, &mai, nullptr, &mem));
    CHECK(vkBindBufferMemory(dev, buffer, mem, 0));
    printf("buffer + memory bound\n");

    // descriptor pool + set
    VkDescriptorPoolSize psz{};
    psz.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; psz.descriptorCount = 1;
    VkDescriptorPoolCreateInfo dpci{};
    dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets = 1; dpci.poolSizeCount = 1; dpci.pPoolSizes = &psz;
    VkDescriptorPool pool;
    CHECK(vkCreateDescriptorPool(dev, &dpci, nullptr, &pool));
    VkDescriptorSetAllocateInfo dsai{};
    dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool = pool; dsai.descriptorSetCount = 1; dsai.pSetLayouts = &dsl;
    VkDescriptorSet dset;
    CHECK(vkAllocateDescriptorSets(dev, &dsai, &dset));

    VkDescriptorBufferInfo dbi{};
    dbi.buffer = buffer; dbi.offset = 0; dbi.range = VK_WHOLE_SIZE;
    VkWriteDescriptorSet wds{};
    wds.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    wds.dstSet = dset; wds.dstBinding = 0; wds.descriptorCount = 1;
    wds.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; wds.pBufferInfo = &dbi;
    vkUpdateDescriptorSets(dev, 1, &wds, 0, nullptr);
    printf("descriptor set updated\n");

    // command pool + buffer
    VkCommandPoolCreateInfo cpci2{};
    cpci2.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpci2.queueFamilyIndex = chosen_family;
    VkCommandPool cpool;
    CHECK(vkCreateCommandPool(dev, &cpci2, nullptr, &cpool));
    VkCommandBufferAllocateInfo cbai{};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = cpool; cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbai.commandBufferCount = 1;
    VkCommandBuffer cmd;
    CHECK(vkAllocateCommandBuffers(dev, &cbai, &cmd));
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    CHECK(vkBeginCommandBuffer(cmd, &bi));
    printf("command buffer began\n");

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    printf("pipeline bound\n");
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &dset, 0, nullptr);
    vkCmdDispatch(cmd, 1, 1, 1);
    printf("dispatched\n");
    CHECK(vkEndCommandBuffer(cmd));
    printf("command buffer recorded\n");

    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
    CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
    CHECK(vkQueueWaitIdle(queue));
    printf("queued + idle\n");

    // read back
    float* host;
    CHECK(vkMapMemory(dev, mem, 0, buf_size, 0, (void**)&host));
    printf("result: ");
    for (uint32_t i = 0; i < N; ++i) printf("%.1f ", host[i]);
    printf("\n");
    vkUnmapMemory(dev, mem);

    printf("PROBE PASSED\n");
    return 0;
}
