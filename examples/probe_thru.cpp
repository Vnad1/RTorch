// Standalone Vulkan throughput probe: initialize ONCE (instance/device/pipeline/
// buffers), then loop dispatch MATMUL_TILES times, timing ONLY the dispatch
// loop. Isolates pure GPU compute time from framework/setup overhead.
// usage: probe_thru <spv> <a> <b> <iters>
#include <vulkan/vulkan.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>

int main(int argc, char** argv) {
    setvbuf(stderr, nullptr, _IONBF, 0);
    if (argc < 5) { fprintf(stderr, "usage: probe_thru <spv> <a> <b> <iters>\n"); return 2; }
    size_t spv_len = 0;
    std::vector<uint8_t> spv = [&]{ std::vector<uint8_t> v; FILE* f=fopen(argv[1],"rb"); if(!f) exit(2); fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET); v.resize(n); fread(v.data(),1,n,f); fclose(f); return v; }();
    std::vector<uint8_t> A, B; { FILE* f=fopen(argv[2],"rb"); fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET); A.resize(n); fread(A.data(),1,n,f); fclose(f); }
    { FILE* f=fopen(argv[3],"rb"); fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET); B.resize(n); fread(B.data(),1,n,f); fclose(f); }
    int iters = atoi(argv[4]);
    const uint32_t N=1024, out_len=N*N*4;

    VkApplicationInfo app{}; app.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO; app.pApplicationName="probe"; app.apiVersion=VK_API_VERSION_1_1;
    VkInstanceCreateInfo ici{}; ici.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO; ici.pApplicationInfo=&app;
    VkInstance inst; if(vkCreateInstance(&ici,0,&inst)!=VK_SUCCESS){return 3;}
    fprintf(stderr,"[probe] instance ok\n");
    VkPhysicalDevice pdev=VK_NULL_HANDLE; uint32_t qfam=0;
    { uint32_t nd=0; vkEnumeratePhysicalDevices(inst,&nd,0); std::vector<VkPhysicalDevice> ds(nd); vkEnumeratePhysicalDevices(inst,&nd,ds.data());
      for(auto pd:ds){uint32_t nq=0; vkGetPhysicalDeviceQueueFamilyProperties(pd,&nq,0); std::vector<VkQueueFamilyProperties> qp(nq); vkGetPhysicalDeviceQueueFamilyProperties(pd,&nq,qp.data());
        for(uint32_t i=0;i<nq;++i) if(qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT){pdev=pd;qfam=i;goto found;} }
      found:; }
    fprintf(stderr,"[probe] device ok qfam=%u\n", qfam);
    float prio=1.0f; VkDeviceQueueCreateInfo dqci{}; dqci.sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO; dqci.queueFamilyIndex=qfam; dqci.queueCount=1; dqci.pQueuePriorities=&prio;
    VkDeviceCreateInfo dci{}; dci.sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO; dci.queueCreateInfoCount=1; dci.pQueueCreateInfos=&dqci;
    VkDevice dev; vkCreateDevice(pdev,&dci,0,&dev); VkQueue queue; vkGetDeviceQueue(dev,qfam,0,&queue);
    fprintf(stderr,"[probe] device+queue ok\n");

    VkPhysicalDeviceMemoryProperties memp{}; vkGetPhysicalDeviceMemoryProperties(pdev,&memp);
    uint32_t dm=0; for(;dm<memp.memoryTypeCount;++dm) if((memp.memoryTypes[dm].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) && !(memp.memoryTypes[dm].propertyFlags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))) break;
    uint32_t hm=0; for(;hm<memp.memoryTypeCount;++hm) if((memp.memoryTypes[hm].propertyFlags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))==(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) break;
    fprintf(stderr,"[probe] mem dm=%u hm=%u count=%u\n", dm, hm, memp.memoryTypeCount);
    if(dm==memp.memoryTypeCount){fprintf(stderr,"no device-local mem\n");return 4;}

    VkShaderModuleCreateInfo smci{}; smci.sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO; smci.codeSize=spv_len; smci.pCode=(const uint32_t*)spv.data();
    VkShaderModule module; vkCreateShaderModule(dev,&smci,0,&module);
    fprintf(stderr,"[probe] shader module ok\n");
    VkDescriptorSetLayoutBinding bindings[3]; for(int i=0;i<3;++i){bindings[i].binding=i;bindings[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;bindings[i].descriptorCount=1;bindings[i].stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;}
    VkDescriptorSetLayoutCreateInfo dsci{}; dsci.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO; dsci.bindingCount=3; dsci.pBindings=bindings;
    VkDescriptorSetLayout dsl; vkCreateDescriptorSetLayout(dev,&dsci,0,&dsl);
    fprintf(stderr,"[probe] dsl ok\n");
    VkPipelineLayoutCreateInfo plci{}; plci.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO; plci.setLayoutCount=1; plci.pSetLayouts=&dsl;
    VkPipelineLayout layout; vkCreatePipelineLayout(dev,&plci,0,&layout);
    fprintf(stderr,"[probe] layout ok\n");
    VkPipelineShaderStageCreateInfo stage{}; stage.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO; stage.stage=VK_SHADER_STAGE_COMPUTE_BIT; stage.module=module; stage.pName="main";
    VkComputePipelineCreateInfo cpci{}; cpci.sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO; cpci.stage=stage; cpci.layout=layout;
    VkPipeline pipeline; vkCreateComputePipelines(dev,0,1,&cpci,0,&pipeline);
    fprintf(stderr,"[probe] pipeline ok\n");

    // device-local buffers for A,B; host staging for upload; device-local output + host staging for readback.
    VkBuffer bufA,bufB,dA,dB,outD,stageOut; VkDeviceMemory mA,mB,mA0,mB0,mOutD,mStOut;
    auto mk=[&](VkDeviceSize sz,uint32_t usage,uint32_t mt,VkBuffer* b,VkDeviceMemory* m){
        VkBufferCreateInfo bci{}; bci.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO; bci.size=sz; bci.usage=usage; bci.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
        vkCreateBuffer(dev,&bci,0,b); VkMemoryRequirements r{}; vkGetBufferMemoryRequirements(dev,*b,&r);
        VkMemoryAllocateInfo mai{}; mai.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO; mai.allocationSize=r.size; mai.memoryTypeIndex=mt;
        vkAllocateMemory(dev,&mai,0,m); vkBindBufferMemory(dev,*b,*m,0); };
    mk(A.size(),VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,hm,&bufA,&mA);      // host A
    mk(B.size(),VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,hm,&bufB,&mB);      // host B
    mk(A.size(),VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT,dm,&dA,&mA0);
    mk(B.size(),VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT,dm,&dB,&mB0);
    mk(out_len,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT,dm,&outD,&mOutD);
    mk(out_len,VK_BUFFER_USAGE_TRANSFER_DST_BIT,hm,&stageOut,&mStOut);

    VkDescriptorPoolSize psz{}; psz.type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; psz.descriptorCount=3;
    VkDescriptorPoolCreateInfo dpci{}; dpci.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO; dpci.maxSets=1; dpci.poolSizeCount=1; dpci.pPoolSizes=&psz;
    VkDescriptorPool pool; vkCreateDescriptorPool(dev,&dpci,0,&pool);
    VkDescriptorSetAllocateInfo dsai{}; dsai.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO; dsai.descriptorPool=pool; dsai.descriptorSetCount=1; dsai.pSetLayouts=&dsl;
    VkDescriptorSet dset; vkAllocateDescriptorSets(dev,&dsai,&dset);
    VkDescriptorBufferInfo dbis[3]={{dA,0,VK_WHOLE_SIZE},{dB,0,VK_WHOLE_SIZE},{outD,0,VK_WHOLE_SIZE}};
    VkWriteDescriptorSet writes[3]; for(int i=0;i<3;++i){writes[i].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;writes[i].dstSet=dset;writes[i].dstBinding=i;writes[i].descriptorCount=1;writes[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;writes[i].pBufferInfo=&dbis[i];}
    vkUpdateDescriptorSets(dev,3,writes,0,0);

    VkCommandPoolCreateInfo cpci2{}; cpci2.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO; cpci2.queueFamilyIndex=qfam;
    VkCommandPool cpool; vkCreateCommandPool(dev,&cpci2,0,&cpool);
    VkCommandBufferAllocateInfo cbai{}; cbai.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO; cbai.commandPool=cpool; cbai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbai.commandBufferCount=1;
    VkCommandBuffer cmd; vkAllocateCommandBuffers(dev,&cbai,&cmd);

    // upload via staging: copy host->device once.
    void* p; vkMapMemory(dev,mA,0,A.size(),0,&p); memcpy(p,A.data(),A.size()); vkUnmapMemory(dev,mA);
    vkMapMemory(dev,mB,0,B.size(),0,&p); memcpy(p,B.data(),B.size()); vkUnmapMemory(dev,mB);

    // single command that copies host->device, then we dispatch each iter by re-recording.
    auto upload=[&](VkCommandBuffer cb){
        VkCommandBufferBeginInfo bi{}; bi.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        vkBeginCommandBuffer(cb,&bi);
        VkBufferCopy c1{0,0,A.size()}; vkCmdCopyBuffer(cb,bufA,dA,1,&c1);
        VkBufferCopy c2{0,0,B.size()}; vkCmdCopyBuffer(cb,bufB,dB,1,&c2);
        vkEndCommandBuffer(cb);
        VkSubmitInfo si{}; si.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO; si.commandBufferCount=1; si.pCommandBuffers=&cb;
        vkQueueSubmit(queue,1,&si,0); vkQueueWaitIdle(queue);
    };
    upload(cmd);

    // prepare a compute command buffer template
    VkCommandBufferBeginInfo bi{}; bi.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    VkCommandBuffer cb2; vkAllocateCommandBuffers(dev,&cbai,&cb2);

    auto run=[&](VkCommandBuffer cb){
        vkBeginCommandBuffer(cb,&bi);
        vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,pipeline);
        vkCmdBindDescriptorSets(cb,VK_PIPELINE_BIND_POINT_COMPUTE,layout,0,1,&dset,0,0);
        vkCmdDispatch(cb,N/16,N/16,1);
        vkEndCommandBuffer(cb);
        VkSubmitInfo si{}; si.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO; si.commandBufferCount=1; si.pCommandBuffers=&cb;
        vkQueueSubmit(queue,1,&si,0); vkQueueWaitIdle(queue);
    };

    auto t0=std::chrono::high_resolution_clock::now();
    for(int i=0;i<iters;++i) run(cb2);
    auto t1=std::chrono::high_resolution_clock::now();
    double ms=std::chrono::duration<double,std::milli>(t1-t0).count()/iters;
    double flops=2.0*N*N*N;
    printf("iters=%d  dispatch avg=%.3f ms  -> %.1f GFLOPS (pure compute, device-local)\n", iters, ms, flops/(ms/1e3)/1e9);
    return 0;
}
