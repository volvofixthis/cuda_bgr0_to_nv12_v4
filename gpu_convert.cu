/*

BGR0 to YUV converter
I have artifacts and changing md5sum now.

*/

// System includes
#include <stdio.h>
#include <time.h>
#include <assert.h>
#include <stdint.h>

// CUDA runtime
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
//#include <helper_functions.h>
//#include <helper_cuda.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void bgr0_to_nv12_pixel(unsigned char *dinput, unsigned char *doutput, int cols, int rows) {
        //int i = threadIdx.x;
        //int j = threadIdx.y;
        //char pixel_data[4];
       
        int col_num = blockIdx.x*blockDim.x+threadIdx.x;
        int row_num = blockIdx.y*blockDim.y+threadIdx.y;
       
        if ((row_num < rows) && (col_num < cols))
    {
       
                //int global_offset = blockIdx.x*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;
                int global_offset = row_num*cols+col_num;
                //char pixel_data[4];
                //memcpy(&pixel_data,dinput+global_offset*4,4);
                int r,g,b;
                r = dinput[global_offset*4+2];
                g = dinput[global_offset*4+1];
                b = dinput[global_offset*4+0];
                //r = pixel_data[2];
                //g = pixel_data[1];
                //b = pixel_data[0];
                doutput[global_offset] = ((66*r + 129*g + 25*b) >> 8) + 16;
                if(((threadIdx.x % 2) == 0)  and ((threadIdx.y % 2) == 0)){
                        doutput[cols*rows+row_num*cols/2+col_num+1] = ((112*r + -94*g + -18*b) >> 8) + 128;
                        doutput[cols*rows+row_num*cols/2+col_num] = ((-38*r + -74*g + 112*b) >> 8) + 128;
                }
       
        }
}

__global__ void bgr0_to_nv12_pixel_uint(unsigned char *dinput, unsigned char *doutput, int cols, int rows) { 
    
    int col_num = blockIdx.x*blockDim.x+threadIdx.x;
    int row_num = blockIdx.y*blockDim.y+threadIdx.y;
    
    if ((row_num < rows) && (col_num < cols)) 
    {
    
        //int global_offset = blockIdx.x*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;
        int global_offset = row_num*cols+col_num; 
        uint32_t a = *((uint32_t *)&dinput[global_offset*4]);
        int r,g,b;
        
        r = a & 0xff;
        g = ( a >> 8 ) & 0xff;
        b = ( a >> 16 ) & 0xff;
        
        
        doutput[global_offset] = ((66*r + 129*g + 25*b) >> 8) + 16;
        if(((threadIdx.x & 1) == 0)  and ((threadIdx.y & 1) == 0)){
            int uv_offset = cols*rows+((row_num*cols)>>1)+col_num;
            doutput[uv_offset] = ((112*r + -94*g + -18*b) >> 8) + 128;
            doutput[uv_offset+1] = ((-38*r + -74*g + 112*b) >> 8) + 128;
        }
    
    }
}

struct cuda_memory_struct
{
   uint8_t *dinput;
   uint8_t *doutput;
};


extern "C" void * cuda_memory_init(int width,int height){
    int pan_size = width*height*4;
    int pan_size_nv12 = width*height/2*3;
    cuda_memory_struct * cuda_memory_ptr = (cuda_memory_struct * )malloc(sizeof(cuda_memory_struct));
    //cuda_memory_ptr.dinput = 
    clock_t end, start;
    float seconds;
    start = clock();
    cudaMalloc((void **)&cuda_memory_ptr->dinput, sizeof(char)*pan_size );
    cudaMalloc((void **)&cuda_memory_ptr->doutput, sizeof(char)*pan_size_nv12 );
    end = clock();
    seconds = (float)(end - start) / CLOCKS_PER_SEC;
    printf("Device malloc: %fs\n",seconds);
    return cuda_memory_ptr;
}

extern "C" int cuda_br0_to_nv12(cuda_memory_struct * cuda_memory_ptr,uint8_t * bgr0,uint8_t * nv12, int width, int height)
{
    int pan_size = width*height*4;
    int pan_size_nv12 = width*height/2*3;
    clock_t end, start;
    float seconds;

    uint8_t *dinput = NULL;
    uint8_t *doutput = NULL;
    dinput = cuda_memory_ptr->dinput;
    doutput = cuda_memory_ptr->doutput;

    start = clock();

    cudaMemcpy(dinput, bgr0, sizeof(unsigned char)*pan_size, cudaMemcpyHostToDevice);
    
    end = clock();
    seconds = (float)(end - start) / CLOCKS_PER_SEC;
    printf("Memory to device copy took: %fs\n",seconds);

    int block_width = 32;
    int block_height = 8;
    int x = width/block_width;
    int y = height/block_height;
    const dim3 numBlocks (x, y, 1);                                // number of blocks 
    const dim3 threadsPerBlock(block_width, block_height, 1);  

    start = clock();

    bgr0_to_nv12_pixel_uint<<<numBlocks, threadsPerBlock>>>(dinput, doutput, width, height);
    gpuErrchk( cudaPeekAtLastError() );
    cudaDeviceSynchronize();

    end = clock();
    seconds = (float)(end - start) / CLOCKS_PER_SEC;
    printf("Convertation using uint took: %fs\n",seconds);
   
    start = clock();

    cudaMemcpy( nv12, doutput, sizeof(unsigned char)*pan_size_nv12, cudaMemcpyDeviceToHost);
    
    end = clock();
    seconds = (float)(end - start) / CLOCKS_PER_SEC;
    printf("Memory to host copy took: %fs\n",seconds);
    
    return 0;
    
}





