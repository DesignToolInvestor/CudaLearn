/*
  K e r n e l . c u
*/

#include <cstdlib>

#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// ****************************************************************************
template<typename ElemT>
__global__ void KernalA(ElemT *data, unsigned size)
{
  unsigned thread = threadIdx.x;
  unsigned elem = (blockIdx.x * gridDim.x) + thread;

  data[elem] += elem;
}

// ************************************
template<typename ElemT>
__global__ void KernalB(ElemT* data, unsigned size)
{
  unsigned thread = threadIdx.x;
  unsigned elem = (blockIdx.x * gridDim.x) + thread;

  if ((elem % 2) == 0)
    data[elem] += elem;
  else
    data[elem] += -elem;
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
template<typename ElemT, void (*Kern)(ElemT* data, unsigned dataElems)>
void InvokeKern(ElemT* data, unsigned dataElems, unsigned threadPerBlock)
{
  ElemT* data_d = NULL;

  // Choose which GPU to run on, change this on a multi-GPU system.
  CheckErr(cudaSetDevice(0), "No cuda devices.");

  // Compute the grid size
  cudaDeviceProp devProp;
  CheckErr(cudaGetDeviceProperties(&devProp, 0), "Can't get device properties");

  // Allocate GPU buffers for data and partSum  
  const size_t dataBytes = dataElems * sizeof(ElemT);
  CheckErr(cudaMalloc((void**)&data_d, dataBytes), "Data allocation failed");

  // Copy input vectors from host memory to GPU buffers.
  CheckErr(cudaMemcpy(data_d, data, dataBytes, cudaMemcpyHostToDevice), "Copying data failed");

  // Launch kernel
  unsigned numThread = dataElems;
  unsigned numBlock = (unsigned)((numThread + (threadPerBlock - 1)) / threadPerBlock);

  Kern<<<numBlock,numThread>>>(data_d, dataElems);

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

  // Copy data back
  CheckErr(
    cudaMemcpy(data, data_d, dataBytes, cudaMemcpyDeviceToHost),
    "Copy of PartSum failed");

  // Clean up
  if (data_d != NULL)
    cudaFree(data_d);
}

// ****************************************************************************
int main()
{
    const unsigned dataSize = 2048;
    const unsigned threadPerBlock = 256;

    float data[dataSize];

    // Initialize data.  This is not sensible, but it demonstrates the use of templates.
    InvokeKern<float, KernalA<float>>(data, dataSize, threadPerBlock);

    // Check the result
    for (unsigned i{ 0 }; i < dataSize; i++)
      if (data[i] != i)
        abort();

    // Initialize data Method A
    // This is not sensible kernel, but it demonstrates the use of templates.
    InvokeKern<float, KernalA<float>>(data, dataSize, threadPerBlock);

    // Check the result
    for (unsigned i{ 0 }; i < dataSize; i++)
      if (data[i] != i)
        abort();

    // Initialize data Method B
    InvokeKern<float, KernalB<float>>(data, dataSize, threadPerBlock);

    // Check the result
    for (unsigned i{ 0 }; i < dataSize; i++)
      if ((((i % 2) == 0) && (data[i] != i)) || (data[i] != -1))
        abort();

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    CheckErr(cudaDeviceReset(), "cudaDeviceReset failed!");

    return 0;
}