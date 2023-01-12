/*
  K e r n e l . c u
*/

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stddef.h>

#include <format>
#include <iostream>
#include <random>

using namespace std;

typedef mt19937_64 CoreGenT;

typedef int ElemT;

// ****************************************************************************
// ToDo:  Won't compile with templates ???
//template<typename ElemT>

__global__ void AddReduceInPlace(
  ElemT* data, unsigned elemPerArray, unsigned elemPerBlock, ElemT* partSum)
{
  // ToDo:  compare speeds with using size_t
  unsigned block = blockIdx.x;
  unsigned blockIndex = block * elemPerBlock;

  unsigned thread = threadIdx.x; // i.e., ofsest within block
  unsigned baseArrayIndex = blockIndex + thread;

  // Thge last block may not be a full array
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

    if (thread == 0)
      printf(
        "Block Num = %d; Block Size = %d; NumThread = %d; Base Index = %d\n",
        (unsigned)block, (unsigned)elemInThisBlock, (unsigned)numActiveThread, (unsigned)baseArrayIndex);

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

      if (thread == 0)
        printf(
          "Block Num = %d; Active Array Size = %d; NumThread = %d\n",
          (unsigned)block, (unsigned)numActiveElem, (unsigned)numActiveThread);

      count++;
    }
  }

  // copy partSum back
  if (thread == 0)
    partSum[block] = data[baseArrayIndex];
}

// ************************************
ElemT ReduceAddCpu(const ElemT* data, size_t size)
{
  ElemT partSum = 0;
  for (size_t i = 0; i < size; i++)
    partSum += data[i];
  return partSum;
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

  cout << "Num Block = " << numBlock << '\n';
  cout << "Elem per Block = " << elemPerBlock << '\n';
  cout << "Tread per Block = " << threadPerBlock << '\n';
  cout << "Actual Number of Thread = " << numBlock * threadPerBlock << "\n\n";

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
  
  // Print poartal sum
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
  constexpr size_t dataSize = 17;
  ElemT data[dataSize] = { 0 };

  // fill data
  constexpr uint64_t seed = 0;
  constexpr float low = 0;
  constexpr float high = 1;

  CoreGenT coreGen(seed);
  uniform_real_distribution<float> dist(low, high);

  //for (ElemT& elem : data)
  //  elem = dist(coreGen);
  for (size_t i = 0; i < dataSize; i++)
    data[i] = i;

  // Add vectors in parallel.
  ElemT resultGpu;

  cudaError_t cudaStatus = ReduceAddGpu(data, dataSize, resultGpu);
  cout << "GPU Result = " << resultGpu << '\n';

  // cudaDeviceReset must be called before exiting in order for profiling and
  // tracing tools such as Nsight and Visual Profiler to show complete traces.
  cudaStatus = cudaDeviceReset();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaDeviceReset failed!");
    return 1;
  }

  // Check result
  ElemT resultCpu = ReduceAddCpu(data, dataSize);
  cout << "CPU Result = " << resultCpu << '\n';

  return 0;
}