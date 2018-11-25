# Matrix Multiplication Example
> 通过串行和并行两种方法，计算矩阵乘矩阵，其中并行方法包括Baseline版本和使用shared_memory优化后的版本，并输出运行结果和运行时间。

## 函数介绍
serialMatmul：串行矩阵相乘函数。

parallelMatMul：利用shared_memory优化后的并行程序的stub函数。

MatrixMulKernel：利用shared_memory优化后的矩阵相乘的kernel。

parallelMatMul_baseline：baseline版本的并行程序的stub函数。

parallelMatMul_baseline：baseline版本的矩阵相乘的kernel。

## 编译运行
编译运行过程同vector_add
