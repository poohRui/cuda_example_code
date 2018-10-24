/**
 * This is a kernel function which mainly deal with the computation in vector add
 *
 * @param A  One of the vector to be add on device
 * @param B  One of the vector to be add on device
 * @param C  The result of vector add
 * @param n  The lenght of the vector
 */
void vecAddKernel(float* A,
                  float* B,
                  float* C,
                  int    n);

/**
 * This function is called stub function which lauching a kernel.
 *
 * @param h_A  One of the vector data will be add.
 * @param h_B  One of the vector data will be add.
 * @param h_C  The result of vector add
 * @param n    The length of the vector
 */
void vecAdd(float* h_A,
            float* h_B,
            float* h_C,
            int    n);




