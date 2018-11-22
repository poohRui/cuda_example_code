#include <stdio.h>
#include <math.h>
#include <cuda.h>

#define BLOCK_DIM 16
#define CHANNELS 3

__global__
void colorToGreyscaleConversion(unsigned char * Pout, 
                                unsigned char * Pin, 
                                int             width, 
                                int             height) {

    int Col = threadIdx.x + blockIdx.x * blockDim.x; 
    int Row = threadIdx.y + blockIdx.y * blockDim.y; 

    if (Col < width && Row < height) {
        int greyOffset = Row*width + Col;
        int rgbOffset = greyOffset*CHANNELS;
        unsigned char r = Pin[rgbOffset ]; 
        Pout[grayOffset] = 0.21f*r + 0.71f*g + 0.07f*b;
    } 
}


void conversionParallel(unsigned char * h_Pout, 
                        unsigned char * h_Pin, 
                        int             width, 
                        int             height){
    
    // Using device parallel calculate the result and finally print the time
    cudaEvent_t start, stop;
    float elapsedTime = 0.0;
    
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    
    unsigned char* d_Pout;
    unsigned char* d_Pin;

    int size = width * height * sizeof(char);

    // Allocates object in the device global memory
    cudaMalloc((void **)&d_Pout, size);
    cudaMalloc((void **)&d_Pin, size);

    // Memory data transfer from host to device
    cudaMemcpy(d_Pin, h_Pin, size, cudaMemcpyHostToDevice);

    // Invoke kernel to do the computation on device
    dim3 dimGrid(ceil(width/(float)(BLOCK_DIM)),ceil(height/(float)(BLOCK_DIM)));
    dim3 dimBlock(BLOCK_DIM,BLOCK_DIM);
    colorToGreyscaleConversion<<<dimGrid, dimBlock>>>(d_Pout, d_Pin, width, height);

    // Transfer back the result from d_Pout to h_Pout
    cudaMemcpy(h_Pout, d_Pout, size, cudaMemcpyDeviceToHost);
    
    // Free device memory for Pout, Pin
    cudaFree(d_Pout);
    cudaFree(d_Pin);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&elapsedTime, start, stop);
    
    printf("Parallel invoke conversion function need %.1fs.\n",elapsedTime);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}