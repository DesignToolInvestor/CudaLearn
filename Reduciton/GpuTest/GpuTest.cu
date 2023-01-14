/*
  G p u T e s t . c u
*/

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stddef.h>

#include <format>
#include <iostream>
#include <random>

// ToDo:  not picking up environment variable
//#define KENNETH_LIB_DIR F:\Users\Kenne.DESKTOP-BT6VROU\Documents\GitHub\MANet-Sim\Lib\KennethLib\KennethLib
//#include "KENNETH_LIB_DIR\SeedManagement.h"

#include "F:\Users\Kenne.DESKTOP-BT6VROU\Documents\GitHub\MANet-Sim\Lib\KennethLib\KennethLib\SeedManagement.h"
#include "F:\Users\Kenne.DESKTOP-BT6VROU\Documents\GitHub\MANet-Sim\Lib\KennethLib\KennethLib\RandSeq.h"

#include "../Library/AddReduceSerial.h"

using namespace std;

// ****************************************************************************
template<typename ElemT>
  __global__ void AddReduceInPlace(
    ElemT* data, unsigned elemPerArray, unsigned elemPerBlock, ElemT* partSum)
{
  // ToDo:  compare speeds with using size_t
  unsigned block = blockIdx.x;
  unsigned blockIndex = block * elemPerBlock;

  unsigned thread = threadIdx.x; // i.e., oftest within block
  unsigned baseArrayIndex = blockIndex + thread;

  // The last block may not be a full array
  unsigned elemInThisBlock = elemPerBlock;
  unsigned numActiveThread = blockDim.x;

  if (block == gridDim.x - 1) {
    elemInThisBlock = elemPerArray - blockIndex;
    numActiveThread = (elemInThisBlock - 1) / 2 + 1;
  }

  // Do this thread's computation
  unsigned otherBlockIndex = thread + numActiveThread;
  if (otherBlockIndex < elemInThisBlock) {
    unsigned otherArrayIndex = baseArrayIndex + numActiveThread;
    data[baseArrayIndex] += data[otherArrayIndex];

    __syncthreads();

    // Higher numbered threads will finish early
    unsigned count = 0;
    while ((thread < numActiveThread) && (1 < numActiveThread)) {
      unsigned numActiveElem = numActiveThread;
      numActiveThread = (numActiveElem + 1) >> 1;

      unsigned otherBlockIndex = thread + numActiveThread;
      if (otherBlockIndex < numActiveElem) {
        unsigned otherArrayIndex = baseArrayIndex + numActiveThread;
        data[baseArrayIndex] += data[otherArrayIndex];
      }
      __syncthreads();

      count++;
    }
  }

  // copy partSum back
  if (thread == 0)
    partSum[block] = data[baseArrayIndex];
}

// ****************************************************************************
template<typename ElemT>
cudaError_t ReduceAddGpu(const ElemT* data, size_t dataSize, ElemT& result)
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
  const unsigned numBlock = 2;
  const unsigned elemPerBlock = (dataSize - 1) / numBlock + 1;
  const unsigned threadPerBlock = (elemPerBlock - 1) / 2 + 1;

  // Allocate GPU buffers for data and partSum  
  const size_t dataBytes = dataSize * sizeof(ElemT);
  cudaStatus = cudaMalloc((void**) &data_d, dataBytes);
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
  AddReduceInPlace <<< numBlock, threadPerBlock >>> (data_d, dataSize, elemPerBlock, partSum_d);

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
  for (unsigned i = 0; i < numBlock; i++)
    cout << "Partial Sum " << i << " = " << partSum[i] << '\n';
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

// ************************************
int main()
{
  constexpr size_t minSize = 1;
  constexpr size_t maxSize = 100;

  for (size_t size{ minSize }; size <= maxSize; size++) {
    int* data = new int[size];
    for (size_t i = 0; i < size; i++)
      data[i] = i;

    int result;
    cudaError_t cudaStatus = ReduceAddGpu<int>(data, size, result);
    delete[] data;

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "cudaDeviceReset failed!");
      return 1;
    }

    if (result != (size - 1) * size / 2) {
      fprintf(stderr, "Got wrong answer!");
      return 1;
    }
  }


  return 0;
}