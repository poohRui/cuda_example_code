CXX=g++

CUDA_INSTALL_PATH=/usr/local/cuda
CFLAGS= -I. -I$(CUDA_INSTALL_PATH)/include `pkg-config --cflags opencv`
LDFLAGS= -L$(CUDA_INSTALL_PATH)/lib64 -lcudart `pkg-config --cflags --libs opencv`

all:
	$(CXX) $(CFLAGS) -c main.cpp -o main.o
	nvcc $(CUDAFLAGS) -c toGrey.cu -o toGrey.o
	$(CXX) main.o toGrey.o -o toGrey $(LDFLAGS) 

clean:
	rm -f *.o toGrey

