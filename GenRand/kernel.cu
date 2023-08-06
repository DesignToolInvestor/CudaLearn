﻿/*
  G e n R a n d . c u
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//#include <stdio.h>

// C++ files
#include <iostream>

using namespace std;

// ************************************
// This is the turbo pascal random number generator.  It is implicitly mod 2^32.
inline __device__ unsigned NextSeedDev(unsigned seed)
{
  const unsigned A = 134'775'813;
  const unsigned B = 1;

  return (A * seed + B);
}

inline unsigned NextSeed(unsigned seed)
{
  const unsigned A = 134'775'813;
  const unsigned B = 1;

  return (A * seed + B);
}

void RandGenCPU(unsigned* result, int masterSeed, int numThread, int randPerThread,
    int numRandNums) {
    unsigned* threadStartSeeds = new unsigned[numThread];
    unsigned startSeedVal = masterSeed;
    unsigned currSeedVal;
    int index;

    for (int j = 0; j < numThread; j++) {
        startSeedVal = threadStartSeeds[j] = NextSeed(startSeedVal);

        index = j * randPerThread;
        result[index] = currSeedVal = startSeedVal;

        for (int i = 1; i < randPerThread; i++) {
            index = (j * randPerThread) + i;
            if (index < numRandNums) {
                currSeedVal = result[index] = NextSeed(currSeedVal);
            }
        }
    }
}

// ************************************
__global__ void RandGenKern(unsigned* results, unsigned* threadSeed, int randPerThread,
    int numRandNums)
{
  int thread = threadIdx.x;

  unsigned rowStart = randPerThread * thread;
  unsigned seed = threadSeed[thread];

  // calculate new loop end (in case we've already reached desired amount of random numbers)
  int loopSize = randPerThread;
  if ((rowStart + randPerThread) > numRandNums) {
      loopSize = numRandNums - rowStart;
  }

  //debug
  printf("thread: %d, loopend: %d, row start: %d, thread seed: %u\n", thread, loopSize, rowStart, seed);

  for (int i = 0; i < loopSize; i++)
    seed = results[rowStart + i] = NextSeedDev(seed);
}

// ************************************
cudaError_t RandGenLaunch(
  unsigned* result, const unsigned masterSeed, const int numThread, const int randPerThread,
    const int numRandNums)
{
  // Need to be declared here because of the goto's
  unsigned* threadSeed = new unsigned[numThread];
  unsigned seed;

  // Verify that we can run on the first CPU 
  // ToDo:  Fix so that it does something sensible on a multi-GPU system.
  cudaError_t cudaStatus = cudaSetDevice(0);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    goto Error;
  }

  // Allocate space on the GPU
  unsigned* resultDev;
  cudaStatus = cudaMalloc((void**)&resultDev, numRandNums * sizeof(unsigned));
  
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed!");
    goto Error;
  }

  unsigned* threadSeedDev;
  cudaStatus = cudaMalloc((void**)&threadSeedDev, numThread * sizeof(unsigned));
  
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed!");
    goto Error;
  }

  // Generate thread seeds ... On the CPU, because it's probably not faster on the GPU.
  seed = threadSeed[0] = masterSeed;
  for (int i{ 1 }; i < numThread; i++)
    seed = threadSeed[i] = NextSeed(seed);

  // Copy thread seed to device
  cudaStatus = cudaMemcpy(
    threadSeedDev, threadSeed, numThread * sizeof(int), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed!");
    goto Error;
  }

  // Launch a kernel on the GPU with one thread for each element.
  RandGenKern <<<1, numThread>>> (resultDev, threadSeedDev, randPerThread, numRandNums);

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
  cudaStatus = cudaMemcpy(result, resultDev, numRandNums * sizeof(int), cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed!");
    goto Error;
  }

Error:
  if (resultDev)
    cudaFree(resultDev);
  if (threadSeedDev)
    cudaFree(threadSeedDev);

  return cudaStatus;
}

// ************************************
int main()
{
  // core parameters
    const int randPerThread = 10;
    const int numRandNums = 101;

    // Derived paramiters
  const int numThread = ((numRandNums - 1) / randPerThread) + 1;
  const unsigned masterSeed = 4;

  // Derived constants
  unsigned result[numRandNums] = { 0 };
  unsigned resultTest[numRandNums] = { 0 }; 

  // call kernel
  cudaError_t status = RandGenLaunch(result, masterSeed, numThread, randPerThread, numRandNums);
  if (status != cudaSuccess)
      cout << "Error code = " << status << "\n";

  // call CPU test
  RandGenCPU(resultTest, masterSeed, numThread, randPerThread, numRandNums);

  // check GPU with test
  for (int i = 0; i < numRandNums; i++) {
      cout << result[i] << " ";
      if (i % 10 == 9) {
          cout << "\n";
      }

      if (result[i] != resultTest[i]) {
          abort();
      }
  }
  cout << "\n";

  cout << "No Error";

  return 0;

}