//
//  main.cpp
//  vector_add
//
//  Created by poohRui on 2018/10/24.
//  Copyright © 2018 poohRui. All rights reserved.
//
#include <iostream>
#include <time.h>
#include <stdlib.h>
#include "vectorAdd.h"
using namespace std;

// Predefine the size of the two vectors
#define N 400

/**
 * This is a function to randomly initial the data in vector
 *
 * @param v  One of the vector to be add on device
 * @param n  The lenght of the vector
 */
void initialVector(float* v,
                   int    n){
    
    for(int i = 0;i<n;i++){
        v[i] = rand() % 100;
    }
}

void vecAddSerial(float* A, float* B, float* C, int n){
    for(int i = 0;i < n;i++){
        C[i] = A[i] + B[i];
    }
}

int main(){
    // Memory allocation for h_A, h_B which contain the vector A and B
    float* h_A = new float[N];
    float* h_B = new float[N];

    // Memory allocation for serial_C and parallel_C which contain the result of serial and parallel
    float* serial_C = new float[N];
    float* parallel_C = new float[N];
    
    // Random initial h_A and h_B, N elements each
    srand(time(NULL));
    initialVector(h_A, N);
    initialVector(h_B, N);
    
    // Invoke the stub funtion(parallel)
    vecAdd(h_A, h_B, parallel_C, N);

    // Show the parallel result
    for(int i = 0;i < N;i++){
        cout<<h_A[i]<<" + "<<h_B[i]<<" = "<<parallel_C[i]<<endl;
    }
    
    // Using the traditional serial code and finally print the time
    clock_t serial_start,serial_end;
    serial_start = clock();
    
    // Invoke the vecAdd serial function
    vecAddSerial(h_A, h_B, serial_C, N);
    
    serial_end = clock();
    double dur = (double)(serial_end - serial_start);
    cout<<"Serial invoke vectorAdd function need "<<dur/CLOCKS_PER_SEC<<"s."<<endl;
    
    // Show the parallel result
    for(int i = 0;i < N;i++){
        cout<<h_A[i]<<" + "<<h_B[i]<<" = "<<serial_C[i]<<endl;
    }
    
}

