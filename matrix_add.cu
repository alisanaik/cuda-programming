#include <iostream>
#include <cmath>

// matrix dimensions
const int N = 10; // 10 * 10 

//kernal func 
//it uses 2d grid and 2d block
//when the func will get called it'll generate many threads and each thread will have their own unique index
__global__ void matrix_addition(int *A, int *B, int *C, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; //col index = index of the block in the grid * number of threads in a block + index of the current thread in the block   
    int j = blockIdx.y * blockDim.y + threadIdx.y; //row index
    
    if (i < N && j < N) { // only 100 elements in the matrix , when the func will get called it'll generate more threads so we need to check the thread index i,j dont exit matrix otherwise it'll try to calculate on non existing element and cause Out of bound memory access error
        int index = j * size + i;
        C[index] = A[index] + B[index];
    }
}

int main() {
    const int matrix_size = N * N * sizeof(int);
    
    // allocate memory on host
    int *h_A = new int[N * N];
    int *h_B = new int[N * N];
    int *h_C = new int[N * N];
    
    // initialize matrices
    for (int i = 0; i < N * N; ++i) {
            h_A[i] = 1;
            h_B[i] = 2;
            h_C[i] = 0;
    }
    
    // allocate memory on device
    int *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, matrix_size);
    cudaMalloc((void**)&d_B, matrix_size);
    cudaMalloc((void**)&d_C, matrix_size);
    
    // copy data from host to device
    cudaMemcpy(d_A, h_A, matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, matrix_size, cudaMemcpyHostToDevice);
    
    // before calling the kernel, define block and grid sizes
    dim3 blockDim(32, 16); // number of threads per block (32*16 = 512 threads per block)
    dim3 gridDim((int)ceil((float)N / blockDim.x), (int)ceil((float)N / blockDim.y)); //represents the blocksize. 10 / 32, 10 / 16 = (1,1) 
    
    // launch kernel
    matrix_addition<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);

     cudaDeviceSynchronize(); // wait for the kernel to finish before accessing the result
    
     // copy result back to host
    cudaMemcpy(h_C, d_C, matrix_size, cudaMemcpyDeviceToHost);
    
    // print result
    std::cout << "Result of matrix addition:" << std::endl;
    for (int i = 0; i < N*N; ++i) {
        std::cout << h_C[i] << ", ";
    }
    
    // free memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return 0;
}