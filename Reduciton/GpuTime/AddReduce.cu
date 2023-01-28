/*
  A d d R e d u c e . c u
*/

#include <iostream>

#include <stdio.h>
#include <stddef.h>

#include <cuda.h>
#include "cuda_runtime.h"

#include "../Library/ReduceAdd.h"
//#include "GridHelper.cuh"
#include "EarlyTerm.cuh"

using namespace std;

// ****************************************************************************
void Ok(cudaError_t status, char* message)
{
  if (status != cudaSuccess) {
    printf(message);
    abort();
  }
}

// ****************************************************************************
template<typename ElemT>
__global__ void WarmingUp(ElemT* partSum, ElemT* data, unsigned dataSize)
{
  unsigned numBlock = blockDim.x;
  int tid = blockIdx.x * numBlock + threadIdx.x;

  partSum[tid % numBlock] = tid;
}

// ****************************************************************************
template<typename ElemT>
void ReduceAddGpu(
  ElemT& result, const ElemT* data, size_t numElem, unsigned threadPerBlock)
{
  ElemT* data_d = NULL;
  ElemT* partSum_d = NULL;

  // Choose which GPU to run on, change this on a multi-GPU system.
  Ok(cudaSetDevice(0), "No cuda devices.");

  // Compute the grid size
  cudaDeviceProp devProp;
  Ok(cudaGetDeviceProperties(&devProp, 0), "Can't get device properties");

  unsigned numThread = (numElem - 1) / 2 + 1;
  unsigned numBlock = (numThread - 1) / threadPerBlock + 1;

  // Allocate GPU buffers for data and partSum  
  const size_t dataBytes = numElem * sizeof(ElemT);
  Ok(cudaMalloc((void**)&data_d, dataBytes), "Data allocation failed");

  const size_t resultBytes = numBlock * sizeof(ElemT);
  Ok(cudaMalloc((void**)&partSum_d, resultBytes), "PartSum allocaiton failed");

  // Copy input vectors from host memory to GPU buffers.
  Ok(cudaMemcpy(data_d, data, dataBytes, cudaMemcpyHostToDevice), "Copying data failed");

  // Create timmers
  cudaEvent_t preWarm, middle, postReduce;
  Ok(cudaEventCreate(&preWarm), "Creation of PreWarm event failed");
  Ok(cudaEventCreate(&middle), "Creation of Middle event failed");
  Ok(cudaEventCreate(&postReduce), "Creation of PostReduce event failed");

  // **********************************
  // Do warm up
  Ok(cudaEventRecord(preWarm), "Recording PreWarm event failed");
  WarmingUp << < numBlock, threadPerBlock >> > (partSum_d, data_d, numElem);

  // compute elapsed time
  Ok(cudaEventRecord(middle), "Recording middle event failed");

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

  float warmTime;
  Ok(cudaEventElapsedTime(&warmTime, preWarm, middle), "Warm up time failed.");
  warmTime *= 1e-3;

  // **********************************
  // Do Add Reduce
  AddReduceEarlyTerm <<< numBlock, threadPerBlock >>> (partSum_d, data_d, numElem);

  // Deal with time
  Ok(cudaEventRecord(postReduce), "Recording PostReduce event failed");

  // Check for any errors launching the kernel
  cudaStatus = cudaGetLastError();
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

  float reduceTime;
  Ok(cudaEventElapsedTime(&reduceTime, middle, postReduce), "reduce time failed");
  reduceTime *= 1e-3;

  printf("%d, %d, %f, %f\n", numElem, threadPerBlock, warmTime, reduceTime);

  // Copy output vector from GPU buffer to host memory.
  ElemT* partSum = new ElemT[numBlock];
  Ok(
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