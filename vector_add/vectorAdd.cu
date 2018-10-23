//
//  vectorAdd.cu
//  vector_add_parallel
//
//  Created by poohRui on 2018/10/23.
//  Copyright © 2018 poohRui. All rights reserved.
//

#include <stdio.h>
#include <iostream>
#include <time.h>
#include <stdlib.h>
#include <cuda.h>
using namespace std;

#define THREAD_DIM 256
#define N 400

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
void vecAdd(float* h_A,
            float* h_B,
            float* h_C,
            int    n){
    
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
    vecAddKernel<<<ceil(n/(float)(THREAD_DIM)), THREAD_DIM>>>(d_A, d_B, d_C, n);
    
    // Transfer back the result from d_C to h_C
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    // Free device memory for A, B, C
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void initialVector(float* h_A,
                   float* h_B,
                   int    n){
    
    srand(time(NULL));
    
    for(int i = 0;i<n;i++){
        h_A[i] = rand() % 100;
        h_B[i] = rand() % 100;
    }
}

int main(){
    // Memory allocation for h_A, h_B, and h_C
    float* h_A = new float[N];
    float* h_B = new float[N];
    float* h_C = new float[N];
    
    // I/O to read h_A and h_B, N elements each
    initialVector(h_A, h_B, N);
    
    // Invoke the stub funtion
    vecAdd(h_A, h_B, h_C, N);
    
    // Show the result
    for(int i = 0;i < N;i++){
        cout<<h_A[i]<<" + "<<h_B[i]<<" = "<<h_C[i]<<endl;
    }
}
