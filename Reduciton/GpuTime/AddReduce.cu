/*
  A d d R e d u c e . c u
*/

#include <iostream>

#include <stdio.h>
#include <stddef.h>

#include <cuda.h>
#include "cuda_runtime.h"

//#include "../Library/ReduceAdd.h"
#include "Timer.cuh"
#include "EarlyTerm.cuh"

using namespace std;

// ****************************************************************************
float ReduceAdd(const float* data, size_t numElem)
{
  float result = 0;
  for (unsigned i = 0; i < numElem; i++)
    result = result + data[i];

  return result;
}

// ****************************************************************************
template<typename ElemT>
__global__ void WarmUp_d(ElemT* partSum, ElemT* inArray, unsigned dataSize)
{
  unsigned numBlock = blockDim.x;
  int tid = (blockIdx.x * numBlock) + threadIdx.x;

  if ((tid % 2) == 0)
    inArray[tid] = tid;
  else
    inArray[tid] = tid;

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
  ElemT* outArray_d, ElemT* inArray_d, size_t numElems, 
  unsigned numBlock, unsigned threadPerBlock)
{
  // launch kernel
  WarmUp_d<ElemT> <<< numBlock, threadPerBlock >>> (outArray_d, inArray_d, numElems);

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
}

// ************************************
template<typename ElemT>
void AddReduceEarlyTerm(
  ElemT* outArray_d, ElemT* inArray_d, size_t numElems,
  unsigned numBlock, unsigned threadPerBlock)
{
  // Launch device code
  AddReduceEarlyTerm<ElemT> <<< numBlock, threadPerBlock >>> (outArray_d, inArray_d, numElems);

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
}

// ****************************************************************************
template<typename ElemT>
void ReduceAddGpu(
  ElemT& result, const ElemT* inArray, size_t origNumElem, unsigned threadPerBlock)
{
  ElemT* inArray_d = NULL;
  ElemT* outArray_d = NULL;

  // Choose which GPU to run on, change this on a multi-GPU system.
  CheckErr(cudaSetDevice(0), "No cuda devices.");

  // Check if the kernel should be called at all.
  if (origNumElem < 2 * threadPerBlock)
    result = ReduceAdd(inArray, origNumElem);
  else {
    // Start the clock
    TickCountT startTicks = ReadTicks_d();

    // Allocate GPU buffers for inArray
    const size_t dataBytes = origNumElem * sizeof(ElemT);
    CheckErr(cudaMalloc((void**)&inArray_d, dataBytes), "Data allocation failed");

    // Copy inArray from host memory to GPU.
    CheckErr(cudaMemcpy(inArray_d, inArray, dataBytes, cudaMemcpyHostToDevice), "Copying inArray failed");

    // Recurse until the result is small enough to do in the CPU
    size_t numElems = origNumElem;
    unsigned numBlock;
    size_t outBytes;

    TickCountT stopTicks[10] = { 0 };
    unsigned level = 0;

    while (2 * threadPerBlock < numElems) {
      // Compute launch parameters
      unsigned numThread = (unsigned)((numElems + 1) >> 1);
      numBlock = (unsigned)((numThread + threadPerBlock - 1) / threadPerBlock);

      // Allocate the output array
      outBytes = numBlock * sizeof(ElemT);
      CheckErr(cudaMalloc((void**)&outArray_d, outBytes), "Result allocation failed");

      // Launch the kernel and wait for synchronization
      AddReduceEarlyTerm(outArray_d, inArray_d, numElems, numBlock, threadPerBlock);
      stopTicks[level++] = ReadTicks_d();

      // Do double-buffering thing
      CheckErr(cudaFree(inArray_d), "CudaFree failed.");
      inArray_d = outArray_d;
      numElems = numBlock;
    }

    // Copy output vector from GPU buffer to host memory.
    ElemT* outArray = new ElemT[numBlock];
    CheckErr(
      cudaMemcpy(outArray, outArray_d, outBytes, cudaMemcpyDeviceToHost),
      "Copy of OutArray failed");
    CheckErr(cudaFree(inArray_d), "CudaFree failed.");

    // Add using CPU
    result = ReduceAdd(outArray, numBlock);
    delete[] outArray;

    // Stop the clock
    printf("%d, %d, ", origNumElem, threadPerBlock);
    for (unsigned i{ 0 }; i < level; i++)
      printf(", % f", TicksToSecs_d(stopTicks[i] - startTicks));
    printf("\n");
  }
}

// ************************************
// Actually create something to like to
template void ReduceAddGpu<float>(
  float& result, const float* inArray, size_t inElemsArg, unsigned threadPerBlock);