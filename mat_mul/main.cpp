//
//  main.cpp
//  mat_mul
//
//  Created by poohRui on 2018/11/21.
//  Copyright © 2018 poohRui. All rights reserved.
//

#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include "MatrixMul.h"
using namespace std;

/**
 * This is a function to randomly initial the data in mat
 *
 * @param mat  One of the vector to be add on device
 * @param m    The number of row in mat
 * @param n    The number of column in mat
 */
void initialMat(float* mat,
                int    m,
                int    n){
    
    for(int i = 0;i<m;i++){
        for(int j = 0;j<n;j++){
            mat[i*n+j] = rand()%10;
        }
    }
}

/**
 * This is a function to compute MatrixMut in serial
 *
 * @param A    Matrix (m,dim)
 * @param B    Matrix (dim,n)
 * @param C    Result Matrix (m,n)
 * @param m    number of row in A
 * @param n    number of column in B
 * @param dim  number of row in B
 */
void serialMatmul(float* A,
                  float* B,
                  float* C, 
                  int    m, 
                  int    n, 
                  int    dim){

    for(int i = 0;i<m;i++){
        for(int j=0;j<n;j++){
            float sum = 0.0;
            for(int d = 0;d<dim;d++){
                sum += A[i*dim+d] * B[d*n+j];
            }
            C[i*n+j] = sum;
        }
    }
}

int main(){
    const int m = 50;
    const int n = 50;
    const int dim = 50;
    float* A = new float[m * dim];
    float* B = new float[dim * n];
    float* serial_C = new float[m * n];
    float* parallel_C = new float[m * n];

    // Initial matrix A(m*dim), matrix B(dim*n)
    srand(time(NULL));
    initialMat(A, m, dim);
    initialMat(B, dim, n);

    // Invoke Serial Calculation
    struct timeval start;
    struct timeval end;
    gettimeofday(&start,NULL);

    serialMatmul(A, B, serial_C, m, n, dim);

    gettimeofday(&end,NULL);
    unsigned long dur = 1000000*(end.tv_sec-start.tv_sec)+ end.tv_usec-start.tv_usec;
    cout<<"Serial invoke vectorAdd function need "<<(double)dur/1000000<<"s."<<endl;

    for(int i = 0;i<10;i++){
        for(int j = 0;j<10;j++){
            cout<<serial_C[i*n+j]<<" ";
        }
        cout<<endl;
    }

    // Invoke Parallel Calculation
    parallelMatMul(A, B, parallel_C, m, n, dim);

    for(int i = 0;i<10;i++){
        for(int j = 0;j<10;j++){
            cout<<parallel_C[i*n+j]<<" ";
        }
        cout<<endl;
    }

}