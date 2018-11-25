#ifndef MATRIX_MUL_H
#define MATRIX_MUL_H

/**
 * This is a baseline stub function of parallel Matmul
 *
 * @param h_A    Matrix (m,dim)
 * @param h_B    Matrix (dim,n)
 * @param h_C    Result Matrix (m,n)
 * @param m      number of row in h_A
 * @param n      number of column in h_B
 * @param dim    number of row in h_B
 */
void parallelMatMul_baseline(float* h_A, 
                             float* h_B, 
                             float* h_C, 
                             int    m, 
                             int    n,
                             int    dim);

/**
 * This is the stub function of parallel Matmul
 *
 * @param h_A    Matrix (m,dim)
 * @param h_B    Matrix (dim,n)
 * @param h_C    Result Matrix (m,n)
 * @param m      number of row in h_A
 * @param n      number of column in h_B
 * @param dim    number of row in h_B
 */
void parallelMatMul(float* h_A, 
                    float* h_B, 
                    float* h_C, 
                    int    m, 
                    int    n,
                    int    dim);

#endif
