//
//  vectorAdd.cu
//  vector_add_parallel
//
//  Created by poohRui on 2018/10/23.
//  Copyright © 2018 poohRui. All rights reserved.
//

#include <stdio.h>
#include <math.h>
#include <cuda.h>

// Predefine the size of block
#define BLOCK_DIM 256

/**
 * This is a kernel function which mainly deal with the computation in vector add
 *
 * @param A  One of the vector to be add on device
 * @param B  One of the vector to be add on device
 * @param C  The result of vector add
 * @param n  The lenght of the vector
 */
__global__
void vecAddKernel(float* A, float* B, float* C, int n){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < n){
        C[i] = A[i] + B[i];
    }
}

/**
 * This function is called stub function which lauching a kernel.
 *
 * @param h_A  One of the vector data will be add.
 * @param h_B  One of the vector data will be add.
 * @param h_C  The result of vector add
 * @param n    The length of the vector
 */
void vecAddParallel(float* h_A,
                    float* h_B,
                    float* h_C,
                    int    n){
    
    // Using device parallel calculate the result and finally print the time
    cudaEvent_t start, stop;
    float elapsedTime = 0.0;
    
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    
    int size = n * sizeof(float);
    float *d_A, *d_B, *d_C;
    
    // Allocates object in the device global memory
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);
    
    // Memory data transfer from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    // Invoke kernel to do the computation on device
    vecAddKernel<<<ceil(n/(float)(BLOCK_DIM)), BLOCK_DIM>>>(d_A, d_B, d_C, n);
    
    // Transfer back the result from d_C to h_C
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    // Free device memory for A, B, C
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&elapsedTime, start, stop);
    
    printf("Parallel invoke vectorAdd function need %.1fs.\n",elapsedTime);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

