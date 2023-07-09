/*
  G e n R a n d . c u
*/

// Old fashion C
#include <stdio.h>

// Cuda Include files
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Parker Library
#include "MappedFile.h"
#include "OldRand.h"

// Parker Cuda Library
#include "../../../Lib/ErrorCheck.cuh"

// ************************************
inline __device__ unsigned RandLoc(unsigned seed)
{
  const unsigned A = 134'775'813;
  const unsigned C = 1;

  return (A * seed + C);
}

// ****************************
__global__ void GenRandFirst(
  unsigned* buff, unsigned* startSeedGpu, const unsigned numThread, const unsigned buffWidth)
{
  unsigned thread = threadIdx.x + blockIdx.x * blockDim.x;

  if (thread < numThread) {
    unsigned* mySeq = buff + (thread * buffWidth);

    mySeq[0] = startSeedGpu[thread];

    for (unsigned i = 1; i < (buffWidth - 1); i++)
      mySeq[i] = RandLoc(mySeq[i - 1]);

    startSeedGpu[thread] = mySeq[buffWidth - 1] = RandLoc(mySeq[buffWidth - 2]);
  }
}

__global__ void GenRandCont(
  unsigned* buff, unsigned *startSeedGpu, const unsigned numThread, const unsigned buffWidth)
{
  unsigned thread = threadIdx.x + blockIdx.x * blockDim.x;

  if (thread < numThread) {
    unsigned* mySeq = buff + (thread * buffWidth);

    mySeq[0] = RandLoc(startSeedGpu[thread]);

    for (unsigned i = 1; i < (buffWidth - 1); i++)
      mySeq[i] = RandLoc(mySeq[i - 1]);

    startSeedGpu[thread] = mySeq[buffWidth - 1] = RandLoc(mySeq[buffWidth - 2]);
  }
}

// ************************************************************
// Setup
// ToDo:  Change to passing a pointer by reference
void GenRandSetUp(
  unsigned** buff, unsigned** startSeedGpu,
  const unsigned buffHeight, const unsigned buffWidth, const unsigned threadPerBlock)
{
  // Choose which GPU to run on, change this on a multi-GPU system.
  CheckErr(cudaSetDevice(0), "No cuda devices");

  // Allocate GPU buffer and startSeedGpu
  const unsigned buffSize = buffHeight * buffWidth * sizeof(unsigned);
  CheckErr(cudaMalloc(buff, buffSize), "Buffer allocation failed.");

  CheckErr(cudaMalloc((void**)&startSeedGpu, buffHeight * sizeof(unsigned)), "StartSeed allocation failed.");
}

// ************************************
// Generate random numbers
void GenRand(
  unsigned result[], const unsigned numRand, const unsigned masterSeed,
  unsigned* buff, unsigned* startSeedGpu,
  const unsigned buffHeight, const unsigned buffWidth, const unsigned threadPerBlock)
{
  // Generate start seed on CPU
  unsigned* startSeedCpu = new unsigned[buffHeight];
  startSeedGpu[0] = masterSeed;
  for (unsigned i = 0; i < buffHeight; i++)
    startSeedCpu[i] = RandTurboPascal(startSeedGpu[i - 1]);

  // Copy start seed to GPU
  CheckErr(
    cudaMemcpy(startSeedGpu, startSeedCpu, buffHeight * sizeof(unsigned), cudaMemcpyHostToDevice),
    "Error loading start seed");

  // Free start seed on CPU
  delete[] startSeedCpu;

  // Compute grid size
  const unsigned randPerBuff = buffHeight * buffWidth;
  const unsigned buffSize = randPerBuff * sizeof(unsigned);
  const unsigned numBuff = (numRand - 1) / randPerBuff + 1;

  const unsigned numBlock = (buffHeight - 1) / threadPerBlock + 1;

  // Launch kernel for first buffer
  GenRandFirst <<<numBlock, threadPerBlock>>> (buff,startSeedGpu, buffHeight,buffWidth);
  CheckErr(cudaGetLastError());

  const unsigned moveSize = max(numRand, buffSize) * sizeof(unsigned);
  CheckErr(
    cudaMemcpy(result, buff, moveSize, cudaMemcpyDeviceToHost),
    "Buffer copy error");

  // Do the kernels for the middle buffers
  for (unsigned buffNum = 1; buffNum < (numBuff - 1); buffNum++) {
    GenRandCont <<<numBlock, threadPerBlock>>> (buff, startSeedGpu, buffHeight, buffWidth);
    CheckErr(cudaGetLastError());

    CheckErr(
      cudaMemcpy(result + randPerBuff * buffNum, buff, buffSize, cudaMemcpyDeviceToHost),
      "Buffer copy error");
  }

  // Do the kernel for the last buffer
  if (numBuff == 1) {
    GenRandCont <<<numBlock, threadPerBlock>>> (buff, startSeedGpu, buffHeight, buffWidth);
    CheckErr(cudaGetLastError());

    const unsigned moveSize = (numRand - numBuff * randPerBuff) * sizeof(unsigned);
    CheckErr(
      cudaMemcpy(result + randPerBuff * (numBuff - 1), buff, moveSize, cudaMemcpyDeviceToHost),
      "Buffer copy error");
  }
}

// ************************************************************
// Shutdown
void GenRandShutDown(unsigned* buff, unsigned* startSeedGpu)
{
  // Wait for last memory copy to finish
  CheckErr(cudaDeviceSynchronize(), "Synchronization failure.");

  cudaFree(buff);
  cudaFree(startSeedGpu);
}

// ************************************************************
int main()
{
  constexpr FileSizeT numRand = 100'000'000;
  constexpr unsigned masterSeed = 100;

  constexpr unsigned buffHeight = 1'000;
  constexpr unsigned buffWidth = 10;
  constexpr unsigned threadPerBlock = 512;

  // Open output file
  const char* fileName = "C:\\Work Space\\Rand Gen.data";
  MappedFileT file(fileName, numRand * sizeof(unsigned));
  unsigned* fileAddr = (unsigned*)file();

  // Generate random numbers
  unsigned* buff;
  unsigned* startSeedGpu;

  GenRandSetUp(&buff, &startSeedGpu, buffHeight, buffWidth, threadPerBlock);

  GenRand(fileAddr, numRand, masterSeed, buff, startSeedGpu, buffHeight, buffWidth, threadPerBlock);
  GenRandShutDown(buff, startSeedGpu);

  // End
  return 0;
};