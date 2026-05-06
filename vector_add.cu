#include<iostream>
#include<cmath>

//host - cpu
//device - gpu

//cpu -> gpu -> cpu
//we're declaring everything in cpu then moving to gpu and then back to cpu for performing operation then getting back the result and storing it in cpu

//first we'll write a kernlal function which will be executed on gpu
// __global__ will turn standard C++ func into a kernel function
__global__ void vector_add(const float* A, const float* B, float* C, int N) {
     
    //calc the index for each thread
    //blockDim.x - number of threads in a block
    //blockIdx.x - block index in the grid
    //threadIdx.x - current thread index within the block
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    // grid -> n. of blocks -> n. of threads
    // there's a grid, each grid has m number of blocks and each block has p number of threads
    // value of no of threads or blocks 256, 512 (typical value) 
    //it wont want to have more threads than the number of elements in the vector, so we need to check if i < N

    //if you dont have this you'll try to access memory out of bounds (OutOfBoundMemoryAccessError)
    if (i < N)
    { 
      C[i] = A[i] + B[i];
    }
    // we didnt write for loop because each thread will execute this code for a different value of i parallel, so we dont need to write a loop, we just need to make sure that we have enough threads to cover all the elements in the vector
}

//now we need to write the main function where we'll make our data trasnfer frpm cpu to device and then call the kernel function and then transfer the result back to cpu

int main(){
    //these  are intialized in cpu
    int const N = 10;
    float A[N],B[N],C[N];

    //Array values for A and B
    for(int i=0; i<N; i++){
        A[i] = (float)i + 1.0f; //initializing the vector A with some values
        B[i] = 2.0f; //initializing the vector B with some values
    }

    //host to device
    //declaration of variable used in Device memory(gpu)
    float *d_a, *d_b, *d_c; //these are pointers variable which will be used in gpu
    
    //we've to allocate memory in gpu for these pointers
    //Allocation of memory in gpu for these pointers
    cudaMalloc(&d_a, N*sizeof(float)); //cudaMalloc is used to allocate memory in gpu, it takes the pointer variable and the size of the memory to be allocated
    cudaMalloc(&d_b, N*sizeof(float));
    cudaMalloc(&d_c, N*sizeof(float));

    //now we need to copy the data from cpu(host) (A and B) to gpu(device) (d_a and d_b)
    cudaMemcpy(d_a, A, N*sizeof(float), cudaMemcpyHostToDevice); //cudaMemcpy is used to copy data from cpu to gpu, it takes the destination pointer, source pointer, size of the data and the direction of the copy
    cudaMemcpy(d_b, B, N*sizeof(float), cudaMemcpyHostToDevice);

    //since weve moved the data to gpu we can now call the kernel function to perform the vector addition on gpu
    //before calling the kernel function configure the kernal dunc which is done by specifying the number of blocks and threads per block
    int blocksize = 256; //number of threads per block
    int gridsize = (int)ceil((float)N/blocksize ); //1
    
    //call the kernel function
    vector_add<<<gridsize, blocksize>>>(d_a,d_b,d_c,N);

    //move your result back to cpu from gpu
    cudaMemcpy(C, d_c, N*sizeof(float), cudaMemcpyDeviceToHost);
    
    
    //print the result
    printf("Vector A: ");
    for(int i=0; i<N; i++){
        std::cout<<A[i]<<(", ");
    }
    printf("\nVector B: ");
    for(int i=0; i<N; i++){
        std::cout<<B[i]<<(", ");
    }   
    printf("\nVector C: ");
    for(int i=0; i<N; i++){
        std::cout<<C[i]<<(", ");
    }

    //since we've allocated tge memory in gpu we need to free it after we're done with it
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}   