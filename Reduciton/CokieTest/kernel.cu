
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#include <device_functions.h>

#include <stdio.h>
#include <stddef.h>

#include <format>
#include <iostream>
#include <random>

//#include "C:\Users\cokie\Workspace\GithubProjects\MANet-Sim\Lib\KennethLib\KennethLib\SeedManagement.h"
//#include "C:\Users\cokie\Workspace\GithubProjects\MANet-Sim\Lib\KennethLib\KennethLib\RandSeq.h"

#include "../Library/AddReduceSerial.h"

// cudaError_t addWithCuda(float *c, const int *a, const int *b, unsigned int size);

using namespace std;

__global__ void AddReduceKernel(float *g_idata, float *g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;

    //covnert global data pointer to the local pointer
    // of this block
    float* idata = g_idata + blockIdx.x * blockDim.x;

    // boundary check
    if (tid >= n) return;

    // in-place reduction in global memory
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if ((tid % (2 * stride)) == 0) {
            idata[tid] += idata[tid + stride];
        }
        // synchronize within block
        __syncthreads();
    }

    //write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];

    //std::cout << "thread " << tid << " result: " << *idata << "\n";
    printf("thread %d result: %f \n", tid, *idata);
    //std::fprintf(stderr, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?\n");
}

//template<typename ElemT>
//cudaError_t ReduceAddGpu(const ElemT* data, size_t dataSize, ElemT& result)
//{
//    ElemT* data_d = NULL;
//    ElemT* partSum_d = NULL;
//    cudaError_t cudaStatus = cudaSuccess;
//
//    // Choose which GPU to run on, change this on a multi-GPU system.
//    cudaStatus = cudaSetDevice(0);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n");
//        goto Error;
//    }
//
//    // Compute gird parameters
//    const unsigned numBlock = 2;
//    const unsigned elemPerBlock = (dataSize - 1) / numBlock + 1;
//    const unsigned threadPerBlock = (elemPerBlock - 1) / 2 + 1;
//
//    // Allocate GPU buffers for data and partSum  
//    const size_t dataBytes = dataSize * sizeof(ElemT);
//    cudaStatus = cudaMalloc((void**)&data_d, dataBytes);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!\n");
//        goto Error;
//    }
//
//    const size_t resultBytes = numBlock * sizeof(ElemT);
//    cudaStatus = cudaMalloc((void**)&partSum_d, resultBytes);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!\n");
//        goto Error;
//    }
//
//    // Copy input vectors from host memory to GPU buffers.
//    cudaStatus = cudaMemcpy(data_d, data, dataBytes, cudaMemcpyHostToDevice);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!\n");
//        goto Error;
//    }
//
//    // Launch a kernel on the GPU with one thread for each element.
//    AddReduceKernel <<< numBlock, threadPerBlock >>> (data_d, dataSize, elemPerBlock, partSum_d);
//
//    // Check for any errors launching the kernel
//    cudaStatus = cudaGetLastError();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
//        goto Error;
//    }
//
//    // cudaDeviceSynchronize waits for the kernel to finish, and returns
//    // any errors encountered during the launch.
//    cudaStatus = cudaDeviceSynchronize();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
//        goto Error;
//    }
//
//    // Copy output vector from GPU buffer to host memory.
//    ElemT partSum[numBlock];
//    cudaStatus = cudaMemcpy(partSum, partSum_d, resultBytes, cudaMemcpyDeviceToHost);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!\n");
//        goto Error;
//    }
//
//    // Print partial sum
//    result = ReduceAddCpu(partSum, numBlock);
//
//    // Print munged data
//    ElemT mungedData[17];
//    cudaStatus = cudaMemcpy(mungedData, data_d, dataBytes, cudaMemcpyDeviceToHost);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!\n");
//        goto Error;
//    }
//
//Error:
//    if (data_d != NULL)
//        cudaFree(data_d);
//    if (partSum_d != NULL)
//        cudaFree(partSum_d);
//
//    return cudaStatus;
//}

template<typename ElemT>
cudaError_t ReduceAddGpu(const float* data, int dataSize, float& result)
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
    const unsigned numBlock = 1;
    const unsigned elemPerBlock = (dataSize - 1) / numBlock + 1;
    const unsigned threadPerBlock = (elemPerBlock - 1) / 2 + 1;

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

    // Launch a kernel on the GPU with one thread for each element.
    AddReduceKernel <<< numBlock, threadPerBlock >>> (data_d, partSum_d, dataSize);

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
    ElemT partSum[numBlock];
    cudaStatus = cudaMemcpy(partSum, partSum_d, resultBytes, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!\n");
        goto Error;
    }

    // Print partial sum
    result = ReduceAddCpu(partSum, numBlock);

    // Print munged data
    ElemT mungedData[17];
    cudaStatus = cudaMemcpy(mungedData, data_d, dataBytes, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!\n");
        goto Error;
    }

Error:
    if (data_d != NULL)
        cudaFree(data_d);
    if (partSum_d != NULL)
        cudaFree(partSum_d);

    return cudaStatus;
}


using namespace std;

// ************************************
int main()
{
    constexpr unsigned startN = 100;
    constexpr unsigned stepPerDec = 8;

    const double nStepFact = exp(log(10) / stepPerDec);

    //const int seed = TimeSeed(6)();
    //RandSeqFloat rand(0, 1, seed);

    // create array to be add-reduced
    int dataSize = 5;
    float* data = new float[dataSize];
    for (int i = 0; i < 5; i++) {
        data[i] = i;
    }

    // add-reduce array
    float result = -1;
    cudaError_t status = ReduceAddGpu<float>(data, dataSize, result);

    // Need a delete [] data
    delete[] data;

    std::cout << result << "\n";
    std::cout << "Hello World!\n";
}