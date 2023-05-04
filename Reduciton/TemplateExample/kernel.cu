/*
  K e r n e l . c u
*/

// Modren C++
#include <cstdlib>
#include <iostream>

// Cuda
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

// ****************************************************************************
template<typename ElemT>
__global__ void KernalA(ElemT *data, unsigned size)
{
  unsigned thread = threadIdx.x;
  unsigned elem = (blockIdx.x * blockDim.x) + thread;

  data[elem] = elem;
}

// ************************************
template<typename ElemT>
__global__ void KernalB(ElemT* data, unsigned size)
{
  unsigned thread = threadIdx.x;
  unsigned elem = (blockIdx.x * blockDim.x) + thread;

  if ((elem % 2) == 0)
    data[elem] = elem;
  else
    data[elem] = -elem;
}

// ****************************************************************************
void CheckErr(cudaError_t status, const char* message)
{
  if (status != cudaSuccess) {
    cout << message;
    abort();
  }
}

// ************************************
// Note:  Typedef doesn't work, but using (which defines an alias) will work.
template<typename ElemT>
using ArrayInitKernT = void (*)(ElemT* data, unsigned dataElems);

template<typename ElemT, typename ArrayInitKernT<ElemT> Kern>
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

  Kern <<<numBlock, threadPerBlock>>>(data_d, dataElems);

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
  typedef int ElemT;

  const unsigned dataSize = 4096;
  const unsigned threadPerBlock = 256;

  ElemT data[dataSize];

  // Initialize data Method A
  // This is not sensible kernel, but it demonstrates the use of templates.
  InvokeKern<ElemT, KernalA<ElemT>>(data, dataSize, threadPerBlock);

  // Check the result
  for (unsigned i{ 0 }; i < dataSize; i++)
    if (data[i] != i)
      abort();

  cout << "Kernal A passed test.\n";

  // Initialize data Method B
  InvokeKern<ElemT, KernalB<ElemT>>(data, dataSize, threadPerBlock);

  // Check the result
  for (unsigned i{ 0 }; i < dataSize; i++)
    if ((((i % 2) == 0) && (data[i] != i)) || (((i % 2) == 1) && (data[i] != -i)))
      abort();

  cout << "Kernal B passed test.\n";

  // cudaDeviceReset must be called before exiting in order for profiling and
  // tracing tools such as Nsight and Visual Profiler to show complete traces.
  CheckErr(cudaDeviceReset(), "cudaDeviceReset failed!");

  return 0;
}