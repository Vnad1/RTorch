import ctypes as C
cl = C.WinDLL("OpenCL.dll")

# prototypes
cl.clGetPlatformIDs.argtypes=[C.c_uint, C.POINTER(C.c_void_p), C.POINTER(C.c_uint)]; cl.clGetPlatformIDs.restype=C.c_int
cl.clGetDeviceIDs.argtypes=[C.c_void_p, C.c_ulonglong, C.c_uint, C.POINTER(C.c_void_p), C.POINTER(C.c_uint)]; cl.clGetDeviceIDs.restype=C.c_int
cl.clGetDeviceInfo.argtypes=[C.c_void_p, C.c_uint, C.c_size_t, C.c_void_p, C.POINTER(C.c_size_t)]; cl.clGetDeviceInfo.restype=C.c_int
cl.clCreateContext.argtypes=[C.POINTER(C.c_int), C.c_uint, C.POINTER(C.c_void_p), C.c_void_p, C.c_void_p, C.POINTER(C.c_int)]; cl.clCreateContext.restype=C.c_void_p
cl.clCreateCommandQueue.argtypes=[C.c_void_p, C.c_void_p, C.c_ulonglong, C.POINTER(C.c_int)]; cl.clCreateCommandQueue.restype=C.c_void_p
cl.clCreateBuffer.argtypes=[C.c_void_p, C.c_ulonglong, C.c_size_t, C.c_void_p, C.POINTER(C.c_int)]; cl.clCreateBuffer.restype=C.c_void_p
cl.clCreateProgramWithSource.argtypes=[C.c_void_p, C.c_uint, C.POINTER(C.c_char_p), C.POINTER(C.c_size_t), C.POINTER(C.c_int)]; cl.clCreateProgramWithSource.restype=C.c_void_p
cl.clBuildProgram.argtypes=[C.c_void_p, C.c_uint, C.POINTER(C.c_void_p), C.c_char_p, C.c_void_p, C.c_void_p]; cl.clBuildProgram.restype=C.c_int
cl.clCreateKernel.argtypes=[C.c_void_p, C.c_char_p, C.POINTER(C.c_int)]; cl.clCreateKernel.restype=C.c_void_p
cl.clSetKernelArg.argtypes=[C.c_void_p, C.c_uint, C.c_size_t, C.c_void_p]; cl.clSetKernelArg.restype=C.c_int
cl.clEnqueueNDRangeKernel.argtypes=[C.c_void_p, C.c_uint, C.POINTER(C.c_size_t), C.POINTER(C.c_size_t), C.POINTER(C.c_size_t), C.c_uint, C.c_void_p, C.c_void_p]; cl.clEnqueueNDRangeKernel.restype=C.c_int
cl.clEnqueueReadBuffer.argtypes=[C.c_void_p, C.c_void_p, C.c_int, C.c_size_t, C.c_size_t, C.c_void_p, C.c_uint, C.c_void_p, C.c_void_p]; cl.clEnqueueReadBuffer.restype=C.c_int
cl.clFinish.argtypes=[C.c_void_p]; cl.clFinish.restype=C.c_int
cl.clGetKernelInfo.argtypes=[C.c_void_p, C.c_uint, C.c_size_t, C.c_void_p, C.POINTER(C.c_size_t)]; cl.clGetKernelInfo.restype=C.c_int

np=C.c_uint()
r=cl.clGetPlatformIDs(0,None,C.byref(np)); print("platforms",r,np.value)
plats=(C.c_void_p*np.value)(); cl.clGetPlatformIDs(np.value,plats,C.byref(np))
dev=None
for p in plats:
    nd=C.c_uint(); cl.clGetDeviceIDs(p,1,0,None,C.byref(nd))
    if not nd.value: continue
    ds=(C.c_void_p*nd.value)(); cl.clGetDeviceIDs(p,1,nd.value,ds,C.byref(nd))
    dev=ds[0]; nm=C.create_string_buffer(256); sz=C.c_size_t()
    cl.clGetDeviceInfo(dev,0x102B,255,nm,C.byref(sz)); print("device",nm.value.decode())
    dv=C.create_string_buffer(256); cl.clGetDeviceInfo(dev,0x0906,255,dv,C.byref(sz)); print("driver_version",dv.value.decode())
    cv=C.create_string_buffer(256); cl.clGetDeviceInfo(dev,0x103D,255,cv,C.byref(sz)); print("opencl_c_version",cv.value.decode())
pv=C.create_string_buffer(256); sz2=C.c_size_t(); cl.clGetPlatformInfo(plats[0],0x0903,255,pv,C.byref(sz2)); print("platform_version",pv.value.decode())
err=C.c_int()
ctx=cl.clCreateContext(None,1,C.byref(C.c_void_p(dev)),None,None,C.byref(err)); print("ctx",ctx,err.value)
q=cl.clCreateCommandQueue(ctx,dev,0,C.byref(err)); print("queue",q,err.value)
g=1; b=C.c_size_t(g*4)
out=cl.clCreateBuffer(ctx,1,b,None,C.byref(err)); print("buf",out,err.value)
src=C.c_char_p(b"__kernel void k(__global float* o){o[get_global_id(0)]=42.0f;}")
pr=cl.clCreateProgramWithSource(ctx,1,C.byref(src),None,C.byref(err)); print("prog",pr,err.value)
r=cl.clBuildProgram(pr,1,C.byref(C.c_void_p(dev)),None,None,None); print("build",r)
k=cl.clCreateKernel(pr,b"k",C.byref(err)); print("kernel",k,err.value)
na=C.c_uint(); cl.clGetKernelInfo(k,0x1191,4,C.byref(na),None); print("numargs",na.value)
r=cl.clSetKernelArg(k,0,8,C.byref(C.c_void_p(out))); print("setarg",r)
gs=C.c_size_t(g); ls=C.c_size_t(1)
r=cl.clEnqueueNDRangeKernel(q,1,None,C.byref(gs),C.byref(ls),0,None,None); print("ndrange",r)
r=cl.clFinish(q); print("finish",r)
h=(C.c_float*1)()
r=cl.clEnqueueReadBuffer(q,out,1,0,b,h,0,None,None); print("read",r,"h0",h[0])
