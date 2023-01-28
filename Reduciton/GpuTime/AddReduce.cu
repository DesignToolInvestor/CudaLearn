/*
  A d d R e d u c e . c u
*/

#include <iostream>

#include <stdio.h>
#include <stddef.h>

#include <cuda.h>
#include "cuda_runtime.h"

#include "../Library/ReduceAdd.h"
#include "Timer.cuh"
#include "EarlyTerm.cuh"

using namespace std;

// ****************************************************************************
template<typename ElemT>
__global__ void WarmUp_d(ElemT* partSum, ElemT* data, unsigned dataSize)
{
  unsigned numBlock = blockDim.x;
  int tid = (blockIdx.x * numBlock) + threadIdx.x;

  if ((tid % 2) == 0)
    data[tid] = tid;
  else
    data[tid] = tid;

  if (tid < numBlock)
    partSum[tid] = tid;

}

// ****************************************************************************
void CheckErr(cudaError_t status, const char* message)
{
  if (status != cudaSuccess) {
    printf(message);
    abort();
  }
}

// ************************************
template<typename ElemT>
void WarmUp(
  ElemT* partSum_d, ElemT* data_d, unsigned numElem, 
  unsigned numBlock, unsigned threadPerBlock, float& time)
{
  cudaEvent_t start, stop;
  CheckErr(cudaEventCreate(&start), "Creation of Start event failed");
  CheckErr(cudaEventCreate(&stop), "Creation of Stop event failed");

  CheckErr(cudaEventRecord(start), "Recording Start event failed");
  WarmUp_d<ElemT> <<< numBlock, threadPerBlock >>> (partSum_d, data_d, numElem);
  CheckErr(cudaEventRecord(stop), "Recording Stop event failed");

  // Check for any errors launching the kernel
  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    abort();
  }

  // Wait for the kernel to finish
  cudaStatus = cudaDeviceSynchronize();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaDeviceSynchronize returned error code %d\n", cudaStatus);
    abort();
  }

  // Compute time
  CheckErr(cudaEventElapsedTime(&time, start, stop), "Elapsed time failed.");
  time *= 1e-3;
}

// ************************************
template<typename ElemT>
void AddReduceEarlyTerm(
  ElemT* partSum_d, ElemT* data_d, unsigned numElem,
  unsigned numBlock, unsigned threadPerBlock, float& time)
{
  // Setup timers
  cudaEvent_t start, stop;
  CheckErr(cudaEventCreate(&start), "Creation of Start event failed");
  CheckErr(cudaEventCreate(&stop), "Creation of Stop event failed");

  // Launch device code
  CheckErr(cudaEventRecord(start), "Recording Start event failed");
  AddReduceEarlyTerm<ElemT> <<< numBlock, threadPerBlock >>> (partSum_d, data_d, numElem);
  CheckErr(cudaEventRecord(stop), "Recording Stop event failed");

  // Check for any errors launching the kernel
  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    abort();
  }

  // Wait for the kernel to finish
  cudaStatus = cudaDeviceSynchronize();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaDeviceSynchronize returned error code %d\n", cudaStatus);
    abort();
  }

  // Compute time
  CheckErr(cudaEventElapsedTime(&time, start, stop), "Elapsed time failed.");
  time *= 1e-3;
}

// ****************************************************************************
template<typename ElemT>
void ReduceAddGpu(
  ElemT& result, const ElemT* data, size_t numElem, unsigned threadPerBlock)
{
  ElemT* data_d = NULL;
  ElemT* partSum_d = NULL;

  // Choose which GPU to run on, change this on a multi-GPU system.
  CheckErr(cudaSetDevice(0), "No cuda devices.");

  // Compute the grid size
  cudaDeviceProp devProp;
  CheckErr(cudaGetDeviceProperties(&devProp, 0), "Can't get device properties");

  unsigned numThread = (numElem - 1) / 2 + 1;
  unsigned numBlock = (numThread - 1) / threadPerBlock + 1;

  // Allocate GPU buffers for data and partSum  
  const size_t dataBytes = numElem * sizeof(ElemT);
  CheckErr(cudaMalloc((void**)&data_d, dataBytes), "Data allocation failed");

  const size_t resultBytes = numBlock * sizeof(ElemT);
  CheckErr(cudaMalloc((void**)&partSum_d, resultBytes), "PartSum allocation failed");

  // Copy input vectors from host memory to GPU buffers.
  CheckErr(cudaMemcpy(data_d, data, dataBytes, cudaMemcpyHostToDevice), "Copying data failed");

  // Do warm up
  float warmTime;
  WarmUp(partSum_d, data_d, numElem, numBlock, threadPerBlock, warmTime);

  // Do add reduce
  TickCountT startTicks = ReadTicks_d();
  float eventTime = 0;
  AddReduceEarlyTerm(partSum_d, data_d, numElem, numBlock, threadPerBlock, eventTime);
  float wallTime = TicksToSecs_d(ReadTicks_d() - startTicks);

  printf("%d, %d, %f, %f\n", numElem, threadPerBlock, eventTime, wallTime);

  // Copy output vector from GPU buffer to host memory.
  ElemT* partSum = new ElemT[numBlock];
  CheckErr(
    cudaMemcpy(partSum, partSum_d, resultBytes, cudaMemcpyDeviceToHost),
    "Copy of PartSum failed");

  result = ReduceAdd(partSum, numBlock);
  delete[] partSum;

  // Clean up
  if (data_d != NULL)
    cudaFree(data_d);
  if (partSum_d != NULL)
    cudaFree(partSum_d);
}

// ************************************
// Actually create something to like to
template void ReduceAddGpu<float>(
  float& result, const float* data, size_t numElem, unsigned threadPerBlock);