Take GT1030 as example.
There are 3 streaming multiprocessors (SM).
There are 128 CUDA cores in each SM, hence 384 CUDA core in total.




Run this to check driver :
>> nvidia-smi

Run this to compile :
>> nncv --version 


* host = CPU
* devices = GPUs
* function run in CUDA is called kernel
* declare __global__ for kernel called by host 
* declare __device__ for kernel called by other kernel in GPU




block CPU until all GPU kernels completed : 

    cudaDeviceSynchronize()

Allocate and free GPU memory : 

    cudaMalloc((void**)&ptr, size);
    cudaFree(ptr);

copy memory from CPU to/from GPU : 

    cudaMemcpy(des, src, size, cudaMemcpyHostToDevice);
    cudaMemcpy(des, src, size, cudaMemcpyDeviceToHost);
    
invoke kernel :

    kernel_name<<<100,256>>>(arg0, arg1, ...); 

    // run this kernel with :
    // 1 grid of 100 blocks, 
    // 1 block of 256 threads, 
    // hence 25600 threads in total

where arg0,1,... are either :
* POD passed by value     <--- which involves copying from CPU to GPU
* pointer to GPU variable <--- which requires cudaMemcpy before calling





What are differences between SM, CUDA core, grid, block, thread?

SM / CUDA core        = physical core
grid / block / thread = logical thread

grid   = collection of blocks for launching a kernel
block  = collection of threads assigned to same SM
thread = assigned to one CUDA core only, but one core can run multiple threads by time-multiplexing
