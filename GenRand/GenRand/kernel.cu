﻿
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void randGenKernel(int* results, int numRandsPerThread,
    int *dev_threadSeeds, const int A, const int B, const int C)
{
    int thread = threadIdx.x;
    int threadSeed = dev_threadSeeds[thread];

    int numRandsSoFar = 0;
    int outputRowIndex = numRandsPerThread * thread;
    int threadCurrSeed = dev_threadSeeds[thread];

    for (int i = 0; i < numRandsPerThread; i++) {
        results[outputRowIndex + i] = (A * threadCurrSeed + B) % C;
    }
}

int main()
{
    const int numThreads = 100;
    const int numRandsPerThread = 100;
    const int masterSeed = 4;

    const unsigned A = 134'775'813;
    const unsigned B = 4;
    const unsigned C = 1;

    int threadSeeds[numThreads] = {0};
    int results[numThreads][numRandsPerThread] = {};

    cudaError_t result;

    // call kernel
    result = RandGenCuda(results, A, C, numThreads, numRandsPerThread, masterSeed, threadSeeds);


    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t RandGenCuda(int** results, const unsigned A, const unsigned C, const int numThreads, const int numRandsPerThread,
    const int masterSeed, int* threadSeeds)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for one vector and one matrix (input, output)    .
    cudaStatus = cudaMalloc((void**)&dev_threadSeeds, numThreads * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, dev_results * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vector from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_threadSeeds, threadSeeds, numThreads * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    randGenKernel<<<1, numThreads>>>(dev_results, numRandsPerThread,dev_threadSeeds, A, B, C);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(results, dev_results, numThreads * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}