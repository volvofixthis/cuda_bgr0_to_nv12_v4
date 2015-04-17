#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <assert.h>
#include <stdint.h>

int main(int argc, char **argv)
{
    int height = 1080;
    int width = 1920;
    int pan_size = width*height*4;
    int pan_size_nv12 = width*height/2*3;
    uint8_t *bgr0;
    uint8_t *nv12;
    bgr0 = (uint8_t *)malloc(sizeof(uint8_t)*pan_size);
    nv12 = (uint8_t *)malloc(sizeof(uint8_t)*pan_size_nv12);
    printf("CUDA bgr0 to nv12\n");
    FILE *ifp;
    ifp = fopen("test.bgr0", "r");
    fread(bgr0,1,pan_size,ifp);
    fclose(ifp);    
    void * cuda_memory_ptr;
    cuda_memory_ptr = (void*)(intptr_t)cuda_memory_init(width,height);
    cuda_br0_to_nv12(cuda_memory_ptr,bgr0,nv12,width,height);
    ifp = fopen("test.nv12", "w");
    fwrite(nv12,1,pan_size_nv12,ifp);
    fclose(ifp);
    return 0;
}

