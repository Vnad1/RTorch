// Compile-time layout check: prints sizeof + key field offsets of Vulkan
// structs using the OFFICIAL vulkan.h, to compare against the hand-written
// #[repr(C)] structs in src/vk.rs.
#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstddef>
#define S(name) printf("%-44s sizeof=%u\n", #name, (unsigned)sizeof(name))
#define OF(name, f) printf("    .%-26s off=%u\n", #f, (unsigned)offsetof(name, f))
int main() {
    S(VkInstanceCreateInfo); OF(VkInstanceCreateInfo, pApplicationInfo);
    S(VkDeviceQueueCreateInfo); OF(VkDeviceQueueCreateInfo, queueFamilyIndex); OF(VkDeviceQueueCreateInfo, pQueuePriorities);
    S(VkDeviceCreateInfo);
    S(VkShaderModuleCreateInfo); OF(VkShaderModuleCreateInfo, codeSize);
    S(VkPipelineShaderStageCreateInfo); OF(VkPipelineShaderStageCreateInfo, module); OF(VkPipelineShaderStageCreateInfo, pName);
    S(VkComputePipelineCreateInfo); OF(VkComputePipelineCreateInfo, layout);
    S(VkDescriptorSetLayoutBinding); OF(VkDescriptorSetLayoutBinding, descriptorType); OF(VkDescriptorSetLayoutBinding, stageFlags);
    S(VkDescriptorSetLayoutCreateInfo);
    S(VkDescriptorPoolSize);
    S(VkDescriptorPoolCreateInfo);
    S(VkDescriptorSetAllocateInfo); OF(VkDescriptorSetAllocateInfo, descriptorPool);
    S(VkWriteDescriptorSet); OF(VkWriteDescriptorSet, dstSet); OF(VkWriteDescriptorSet, descriptorType); OF(VkWriteDescriptorSet, pBufferInfo);
    S(VkDescriptorBufferInfo); OF(VkDescriptorBufferInfo, buffer);
    S(VkDescriptorImageInfo);
    S(VkBufferCreateInfo); OF(VkBufferCreateInfo, size);
    S(VkMemoryRequirements);
    S(VkMemoryAllocateInfo);
    S(VkMemoryType); S(VkMemoryHeap); S(VkPhysicalDeviceMemoryProperties); OF(VkPhysicalDeviceMemoryProperties, memoryTypes); OF(VkPhysicalDeviceMemoryProperties, memoryHeaps);
    S(VkQueueFamilyProperties);
    S(VkCommandPoolCreateInfo);
    S(VkCommandBufferAllocateInfo); OF(VkCommandBufferAllocateInfo, level);
    S(VkCommandBufferBeginInfo);
    S(VkMemoryBarrier); OF(VkMemoryBarrier, srcAccessMask);
    S(VkSubmitInfo);
    return 0;
}
