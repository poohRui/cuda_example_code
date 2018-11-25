#include<cuda.h>
#include<stdio.h>
#include<math.h>

#define TILE_WIDTH 32

/**
 * This is a kernel MatrixMul function of parallel Matmul
 *
 * @param A    Matrix (m,dim)
 * @param B    Matrix (dim,n)
 * @param C    Result Matrix (m,n)
 * @param m      number of row in h_A
 * @param n      number of column in h_B
 * @param dim    number of row in h_B
 */
__global__
void MatrixMulKernel(float* A, 
                     float* B, 
                     float* C,
                     int    m, 
                     int    n, 
                     int    dim){

    __shared__ float Ads[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; 
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Identify the row and column of the C element to work on
    // Each thread works on an element of C
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    // Calculate the number of phase
    int phase_num = ceil(dim / (float)TILE_WIDTH);

    bool cond0 = Row < m;
    bool cond1 = Col < n;

    float Cvalue = 0;
    // Each thread loads 'Row'th row of A and 'Col'th column of B
    for (int ph = 0; ph < phase_num; ++ph) {
        
        if(ph * TILE_WIDTH + tx < dim){
            Ads[ty][tx] = (cond0)?A[Row * dim + ph*TILE_WIDTH + tx]:0;   
        }
        else{
            Ads[ty][tx] = 0;
        }
        if(ph * TILE_WIDTH + ty < dim){
            Bds[ty][tx] = (cond1)?B[(ph*TILE_WIDTH + ty)*n + Col]:0;
        }
        else{
            Bds[ty][tx] = 0;
        }

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) { 
            Cvalue += Ads[ty][k] * Bds[k][tx];
        }

        __syncthreads();
    }

    if(cond0 && cond1){
        C[Row * n + Col] = Cvalue;
    }
}

/**
 * This is a parallel Stub function of parallel Matmul
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
                    int    dim){

    // Using device parallel calculate the result and finally print the time
    cudaEvent_t start, stop;
    float elapsedTime = 0.0;
    
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    float *d_A, *d_B, *d_C;

    size_t size_of_float = sizeof(float);
    size_t size_A = m*dim*size_of_float;
    size_t size_B = n*dim*size_of_float;
    size_t size_C = m*n*size_of_float;

    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_B, size_B);
    cudaMalloc((void**)&d_C, size_C);
    
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);


    // Invoke kernel
    dim3 dimGrid(ceil(n/(float)(TILE_WIDTH)),ceil(m/(float)(TILE_WIDTH)),1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH,1);
    MatrixMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, m, n, dim);

    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);
    
    // Free device memory for A, B, C
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start, stop);

    printf("Parallel invoke Matmul function need %.1fs.\n",elapsedTime);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

/**
 * This is a baseline kernel function of parallel Matmul
 *
 * @param A    Matrix (m,dim)
 * @param B    Matrix (dim,n)
 * @param C    Result Matrix (m,n)
 * @param m      number of row in h_A
 * @param n      number of column in h_B
 * @param dim    number of row in h_B
 */
__global__
void MatrixMulKernel_Baseline(float* A, 
                              float* B, 
                              float* C,
                              int    m, 
                              int    n, 
                              int    dim){
    
    int bx = blockIdx.x; 
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
                            
    // Identify the row and column of the C element to work on
    // Each thread works on an element of C
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    bool cond0 = Row < m;
    bool cond1 = Col < n;

    if(cond0 && cond1){
        float Cvalue = 0;
        for(int i = 0;i<dim;i++){
            Cvalue += A[Row*dim+i]*B[i*n+Col];
        }
        C[Row*n+Col] = Cvalue;
    }
}

/**
 * This is a baseline Parallel Stub function of parallel Matmul
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
                             int    dim){

    // Using device parallel calculate the result and finally print the time
    cudaEvent_t start, stop;
    float elapsedTime = 0.0;
    
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    float *d_A, *d_B, *d_C;

    size_t size_of_float = sizeof(float);
    size_t size_A = m*dim*size_of_float;
    size_t size_B = n*dim*size_of_float;
    size_t size_C = m*n*size_of_float;

    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_B, size_B);
    cudaMalloc((void**)&d_C, size_C);
    
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);


    // Invoke kernel
    dim3 dimGrid(ceil(n/(float)(TILE_WIDTH)),ceil(m/(float)(TILE_WIDTH)),1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH,1);
    MatrixMulKernel_Baseline<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, m, n, dim);

    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);
    
    // Free device memory for A, B, C
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start, stop);

    printf("The baseline Parallel invoke Matmul function need %.1fs.\n",elapsedTime);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}