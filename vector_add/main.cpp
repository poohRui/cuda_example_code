//
//  main.cpp
//  vector_add
//
//  Created by poohRui on 2018/10/24.
//  Copyright © 2018 poohRui. All rights reserved.
//
#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include "vectorAdd.h"
using namespace std;

// Predefine the size of the two vectors
#define N 400

/**
 * This is a function to randomly initial the data in vector
 *
 * @param v  The vector that will be initial randomly
 * @param n  The length of the vector
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
    vecAddParallel(h_A, h_B, parallel_C, N);

    // Show the parallel result
    for(int i = 0;i < 10;i++){
        cout<<h_A[i]<<" + "<<h_B[i]<<" = "<<parallel_C[i]<<endl;
    }
    
    // Using the traditional serial code and finally print the time
    struct timeval start;
    struct timeval end;
    gettimeofday(&start,NULL);
    
    // Invoke the vecAdd serial function
    vecAddSerial(h_A, h_B, serial_C, N);
    
    gettimeofday(&end,NULL);
    unsigned long dur = 1000000*(end.tv_sec-start.tv_sec)+ end.tv_usec-start.tv_usec;
    cout<<"Serial invoke vectorAdd function need "<<(double)dur/1000000<<"s."<<endl;
    
    // Show the parallel result
    for(int i = 0;i < 10;i++){
        cout<<h_A[i]<<" + "<<h_B[i]<<" = "<<serial_C[i]<<endl;
    }
    
}

