//
//  main.cpp
//  mat_mul
//
//  Created by poohRui on 2018/11/21.
//  Copyright © 2018 poohRui. All rights reserved.
//

#include <iostream>
#include <time.h>
#include <stdlib.h>
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
            mat[i*n+j] = rand()%100;
        }
    }
}

float* serialMatmul(float* A,float* B,int m, int n, int dim){
    float* C = new float[m*n];
    for(int i = 0;i<m;i++){
        for(int j=0;j<n;j++){
            float sum = 0.0;
            for(int d = 0;d<dim;d++){
                sum = A[i*dim+d] * B[dim*n+j];
            }
            C[i*n+j] = sum;
        }
    }
}

int main(){
    const int m = 10;
    const int n = 10;
    const int dim = 5;
    float* A = new float[m * dim];
    float* B = new float[dim * n];

    // Initial matrix A(m*dim), matrix B(dim*n)
    srand(time(NULL));
    initialMat(A, m, dim);
    initialMat(B, dim, n);

    // 

}