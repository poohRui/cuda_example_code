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

#define BLOCK_DIM 256
#define N 400

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
    vecAddKernel<<<ceil(n/(float)(BLOCK_DIM)), BLOCK_DIM>>>(d_A, d_B, d_C, n);
    
    // Transfer back the result from d_C to h_C
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    // Free device memory for A, B, C
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

/**
 * This is a function to randomly initial the data in vector A and vector B
 *
 * @param A  One of the vector to be add on device
 * @param B  One of the vector to be add on device
 * @param n  The lenght of the vector
 */
void initialVector(float* h_A,
                   float* h_B,
                   int    n){
    
    srand(time(NULL));
    
    for(int i = 0;i<n;i++){
        h_A[i] = rand() % 100;
        h_B[i] = rand() % 100;
    }
}

void vecAddSerial(float* A, float* B, float* C, int n){
    for(int i = 0;i < n;i++){
        C[i] = A[i] + B[i];
    }
}

int main(){
    // Memory allocation for h_A, h_B, and h_C
    float* h_A = new float[N];
    float* h_B = new float[N];
    float* h_C = new float[N];
    
    // I/O to read h_A and h_B, N elements each
    initialVector(h_A, h_B, N);
    
    // Using device parallel calculate the result and finally print the time
    cudaEvent_t start, stop;
    float elapsedTime = 0.0;
    
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    
    // Invoke the stub funtion(parallel)
    vecAdd(h_A, h_B, h_C, N);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&elapsedTime, start, stop);
    
    cout<<"Parallel invoke vectorAdd function need "<<elapsedTime<<"s."<<endl;
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    // Using the traditional serial code and finally print the time
    clock_t serial_start,serial_end;
    serial_start = clock();
    
    // Invoke the vecAdd serial function
    vecAddSerial(h_A, h_B, h_C, N);
    
    serial_end = clock();
    double dur = (double)(serial_end - serial_start);
    cout<<"Serial invoke vectorAdd function need "<<dur/CLOCKS_PER_SEC<<"s."<<endl;
    
    // Show the result
    /* for(int i = 0;i < N;i++){
        cout<<h_A[i]<<" + "<<h_B[i]<<" = "<<h_C[i]<<endl;
    }*/
}
