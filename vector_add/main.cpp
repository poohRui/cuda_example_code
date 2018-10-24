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

#define N 400

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
    
    // Invoke the stub funtion(parallel)
    vecAdd(h_A, h_B, h_C, N);
    
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

