/*
  G e n R a n d . c u
*/


// C++ filescb
#include <iostream>

// Cuda specific files
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// ParkerLib Files
#include "C:\Users\cokie\Workspace\GithubProjects\ParkerLib\Timer.h"
//#include "Timer.h"
//#include "MappedFile.h"
#include <windows.h>
#include <cassert>

using namespace std;
typedef unsigned long long SizeT;

//***********************************************************************************************
TickCountT ReadTicks() {
    LARGE_INTEGER result;
    int status = QueryPerformanceCounter(&result);
    assert(status != 0);

    return (TickCountT)result.QuadPart;
} // ReadTicks

float TicksToSecs(TickCountT ticks) {
    LARGE_INTEGER freq;
    int status = QueryPerformanceFrequency(&freq);
    assert(status != 0);

    return float(ticks) / float(freq.QuadPart);
} // TicksToSec

//***********************************************************************************************
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

void RandGenCPU(unsigned* result, int masterSeed, SizeT numThread, int randPerThread,
    SizeT numRandNums) {
    unsigned* threadStartSeeds = new unsigned[numThread];
    unsigned startSeedVal = masterSeed;
    unsigned currSeedVal;
    SizeT index;

    for (SizeT j = 0; j < numThread; j++) {
        startSeedVal = threadStartSeeds[j] = NextSeed(startSeedVal);

        index = j * randPerThread;
        result[index] = currSeedVal = startSeedVal;

        for (SizeT i = 1; i < randPerThread; i++) {
            index = (j * randPerThread) + i;
            if (index < numRandNums) {
                currSeedVal = result[index] = NextSeed(currSeedVal);
            }
        }
    }
}

//***********************************************************************************************
__global__ void RandGenKern(unsigned* results, unsigned* threadSeed, int randPerThread,
    SizeT numRandNums)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

  unsigned rowStart = randPerThread * idx;
  unsigned seed = threadSeed[idx];

  // calculate new loop end (in case we've already reached desired amount of random numbers)
  int loopSize = randPerThread;
  if ((rowStart + randPerThread) > numRandNums) {
      loopSize = numRandNums - rowStart;
  }

  //debug
  // printf("thread: %d, loopend: %d, row start: %d, thread seed: %u\n", thread, loopSize, rowStart, seed);

  for (int i = 0; i < loopSize; i++)
    seed = results[rowStart + i] = NextSeedDev(seed);
}

//***********************************************************************************************
cudaError_t RandGenLaunch(
  unsigned* result, const unsigned masterSeed, const SizeT numThread, const int randPerThread,
    const SizeT numRandNums, const int numBlocks, const int threadsPerBlock)
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
  RandGenKern <<<numBlocks, threadsPerBlock>>> (resultDev, threadSeedDev, randPerThread, numRandNums);

  // Check for any errors launching the kernel
  cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "RandGenKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
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

//***********************************************************************************************
int main()
{
  // core parameters
  const int randPerThread = 10;
  const SizeT numRandNums = 4'000'000'000ull;

  // Derived parameters
  const SizeT numThread = ((numRandNums - 1) / randPerThread) + 1;
  cout << "numThread: " << numThread << "\n";
  const unsigned masterSeed = 4;

  // decide on number of threads per block
  const int threadsPerBlock = 512;
  int numBlocks = (numThread - 1) / threadsPerBlock + 1;

  // Derived constants
  unsigned *result = new unsigned[numRandNums];
  unsigned *resultTest = new unsigned[numRandNums];

  //*****************************

  // call CPU test
  RandGenCPU(resultTest, masterSeed, numThread, randPerThread, numRandNums);

  //*****************************

  // check GPU with test
  TickCountT start = ReadTicks();

  // call kernel
  cudaError_t status = RandGenLaunch(result, masterSeed, numThread,
      randPerThread, numRandNums, numBlocks, threadsPerBlock);
  if (status != cudaSuccess)
    cout << "Error code = " << status << "\n";

  // Print elapsed time for test
  TickCountT stop = ReadTicks();
  float seconds = TicksToSecs(stop - start);

  //*****************************

  for (int i = 0; i < numRandNums; i++) {
    // cout << result[i] << " ";
    // if (i % 10 == 9)
    //  cout << "\n";

    if (result[i] != resultTest[i])
      abort();
  }
  // cout << "\n";

  cout << "Elapsed seconds for test: " << seconds << "\n";

  // Exit

  cout << "No Error";

  return 0;
}