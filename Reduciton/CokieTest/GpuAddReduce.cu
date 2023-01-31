#include "GpuAddReduce.h"
#include "UtilMiscCokie.h"

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stddef.h>

#include <format>
#include <iostream>
#include <random>


__global__ void AddReduceKernel(float* g_idata, float* g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;

    //covnert global data pointer to the local pointer
    // of this block
    float* block = g_idata + blockIdx.x * blockDim.x;

    // boundary check
    if (tid >= n) return;

    // in-place reduction in global memory
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (((tid % (2 * stride)) == 0) && ((tid + stride) < blockDim.x)) {
            block[tid] += block[tid + stride];
        }

        //debug
        /*if (tid == 0) {
            printf("stride: %d , blockSize: %d\n", stride, blockDim.x);
        }*/

        // synchronize within block
        __syncthreads();
    }

    //write result for this block to global mem
    if (tid == 0) {
        g_odata[blockIdx.x] = block[0];

        //debug
        /*printf("thread %d result: %f \n", tid, *block);*/
    }
}

template<typename ElemT>
cudaError_t ReduceAddGpu(const ElemT* data, int dataSize, ElemT& result)
{
    ElemT* data_d = NULL;
    ElemT* partSum_d = NULL;
    cudaError_t cudaStatus = cudaSuccess;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n");
        goto Error;
    }

    // Compute gird parameters
    const unsigned elemPerBlock = 1536/2;
    const unsigned numBlock = ((dataSize - 1) / elemPerBlock) + 1;
    const unsigned threadPerBlock = elemPerBlock;

    // Allocate GPU buffers for data and partSum  
    const size_t dataBytes = dataSize * sizeof(ElemT);
    cudaStatus = cudaMalloc((void**)&data_d, dataBytes);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!\n");
        goto Error;
    }

    const size_t resultBytes = numBlock * sizeof(ElemT);
    cudaStatus = cudaMalloc((void**)&partSum_d, resultBytes);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!\n");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(data_d, data, dataBytes, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!\n");
        goto Error;
    }

    // timing code
    TickCountT start_ticks = ReadTicks();

    // Launch a kernel on the GPU with one thread for each element.
    AddReduceKernel << < numBlock, threadPerBlock >> > (data_d, partSum_d, dataSize);

    // timing code
    cudaDeviceSynchronize();
    TickCountT end_ticks = ReadTicks();
    float time_elapsed = TicksToSecs(end_ticks - start_ticks);
    printf("%f %d\n", time_elapsed, dataSize);

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
    ElemT* partSum = new ElemT[numBlock];
    cudaStatus = cudaMemcpy(partSum, partSum_d, resultBytes, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!\n");
        goto Error;
    }

    // Print partial sum
    result = ReduceAdd(partSum, numBlock);

    // Print munged data
    ElemT* mungedData = new float[dataSize];
    cudaStatus = cudaMemcpy(mungedData, data_d, dataBytes, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!\n");
        goto Error;
    }

    // debug
    /*std::cout << "\n";
    for (int i = 0; i < dataSize; i++) {
        std::cout << mungedData[i] << ", ";
    }
    std::cout << "\n";*/

Error:
    if (data_d != NULL)
        cudaFree(data_d);
    if (partSum_d != NULL)
        cudaFree(partSum_d);

    return cudaStatus;
}

template cudaError_t ReduceAddGpu<float>(const float* data, int dataSize, float& result);