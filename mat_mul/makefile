.PHONY: all MatrixMul

all: MatrixMul

MatrixMul:
	nvcc -Wno-deprecated-gpu-targets -o MatrixMul main.cpp MatrixMul.cu
clean:
	rm MatrixMul
