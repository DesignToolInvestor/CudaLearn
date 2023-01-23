/*
  G p u T e s t . c u
*/


#include <iostream>

#include <stdio.h>
#include <stddef.h>

#include <cuda.h>
#include "cuda_runtime.h"

#include "../Library/ReduceAdd.h"
#include "GridHelper.cuh"
#include "EarlyTerm.cuh"

using namespace std;

// ****************************************************************************
template<typename ElemT>
cudaError_t ReduceAddGpu(
  ElemT& result, const ElemT* data, size_t numElem, unsigned threadPerBlock)
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

  // Compute the grid size
  cudaDeviceProp devProp;
  cudaStatus = cudaGetDeviceProperties(&devProp, 0);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "Didn't get device properties!\n");
    goto Error;
  }

  unsigned numThread = (numElem - 1) / 2 + 1;
  dim3 grid = GridSizeSimple(numThread, threadPerBlock, devProp);
  unsigned numBlock = grid.x * grid.y * grid.z;

  // Allocate GPU buffers for data and partSum  
  const size_t dataBytes = numElem * sizeof(ElemT);
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
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  AddReduceEarlyTerm <<< grid, threadPerBlock >>> (partSum_d, data_d, numElem);

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

  cudaEventRecord(stop);
  float time = cudaEventElapsedTime(&time, start,stop);
  time *= 1e3;

  cout << numElem << ", " << threadPerBlock << ", " << time << '\n';

  // Copy output vector from GPU buffer to host memory.
  ElemT* partSum = new ElemT[numBlock];
  cudaStatus = cudaMemcpy(partSum, partSum_d, resultBytes, cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed!\n");
    goto Error;
  }
  delete[] partSum;
  
  // Print partial sum
  result = ReduceAdd(partSum, numBlock);

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
//int main()
//{
//  constexpr size_t minSize = 15;
//  constexpr size_t maxSize = 17;
//  constexpr unsigned threadPerBlock = 8;
//
//  for (size_t size{ minSize }; size <= maxSize; size++) {
//    int* data = new int[size];
//    for (size_t i = 0; i < size; i++)
//      data[i] = i;
//
//    int result;
//    cudaError_t cudaStatus = ReduceAddGpu<int>(result, data, size, threadPerBlock);
//    delete[] data;
//
//    cudaStatus = cudaDeviceReset();
//    if (cudaStatus != cudaSuccess) {
//      fprintf(stderr, "cudaDeviceReset failed!");
//      return 1;
//    }
//
//    if (result != (size - 1) * size / 2) {
//      fprintf(stderr, "Got wrong answer!");
//      return 1;
//    } else
//      fprintf(stderr, "Size = %d passed\n", size);
//  }
//
//  return 0;
//}

// Actually create something to like to
template cudaError_t ReduceAddGpu<int>(
  int& result, const int* data, size_t numElem, unsigned threadPerBlock);