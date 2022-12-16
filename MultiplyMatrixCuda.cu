#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <assert.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char* file, int line, bool abort=true){
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

__global__ void kernel(int* matrix, int* res_matrix, int size){
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int column = blockDim.y * blockIdx.y + threadIdx.y;
    if (row < size && column < size){
        int sum = 0;
        for (int rank = 0; rank < size; rank++)
            sum += matrix[row * size + rank] * matrix[rank * size + column];
        res_matrix[row * size + column] = sum;
    }
}
extern "C"  void multiplyMatrix(int* matrix, int* res_matrix, int size){
    int* cuda_matrix;
    gpuErrchk( cudaMalloc((void**)&cuda_matrix, size * size * sizeof(int)) );
    int* cuda_res_matrix;
    gpuErrchk( cudaMalloc((void**)&cuda_res_matrix, size * size * sizeof(int)) );
    gpuErrchk( cudaMemcpy(cuda_matrix, matrix, size * size * sizeof(int), cudaMemcpyHostToDevice) );
    dim3 threadsPerBlock(size, size);
    dim3 numBlocks(size / threadsPerBlock.x, size / threadsPerBlock.y);
    kernel<<<numBlocks, threadsPerBlock>>>(cuda_matrix, cuda_res_matrix, size);
    gpuErrchk( cudaMemcpy(res_matrix, cuda_res_matrix, size * size * sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk( cudaFree(cuda_matrix) );
    gpuErrchk( cudaFree(cuda_res_matrix) );
}
