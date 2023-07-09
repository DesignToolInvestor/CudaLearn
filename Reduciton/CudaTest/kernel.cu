
#include "cuda_runtime.h"
//#include "device_launch_parameters.h"

#include <stdio.h>


int main()
{
  cudaError_t cudaStatus;
  constexpr unsigned devNum = 0;

  cudaStatus = cudaSetDevice(devNum);
  if (cudaStatus != cudaSuccess)
    fprintf(stderr, "cudaSetDevice failed!");

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, devNum);

  // cudaDeviceReset must be called before exiting in order for profiling and tracing tools (such 
  // as Nsight and Visual Profiler) to show complete traces.
  cudaStatus = cudaDeviceReset();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaDeviceReset failed!");
    return 1;
  }

  return 0;
}
