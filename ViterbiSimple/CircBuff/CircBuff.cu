/*
  C i r c B u f f . c u
*/

// Modern C++
#include <cstdlib>
#include <iostream>

// Old fashion C
#include <stdio.h>

// Cuda
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Local code units
#include "../../Lib/ErrorCheck.cuh"

// ToDo: change to using the environment variable ParkerLibDir
#include "../../../ParkerLib/ParkerLib/OldRand.h"
#include "../../../ParkerLib/ParkerLib/MappedFile.h"

// ******************************
cudaError_t SetUpDev(unsigned** buff, unsigned numSeed, unsigned threadPerBlock)
{
  // Choose which GPU to run on, change this on a multi-GPU system.
  cudaStatus = cudaSetDevice(0);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    goto Error;
  }

  cudaStatus = cudaMalloc((void**)buff, numSeed * sizeof(unsigned));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed!");
    goto Error;
  }
}

// ******************************
__global__ void GenBuff(unsigned* buff, unsigned numSeed)
{
  unsigned index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < numSeed)
    buff[index] = RandTurboPascal(buff[index]);
}

// ****************************************************************************
int main()
{
  constexpr unsigned masterSeed = 0;

  constexpr unsigned numSeq = 1'000'000;

  constexpr unsigned numSeed = 1'024;
  constexpr unsigned threadPerBlock = 32;

  // **********************************
  unsigned seed[numSeed];
  seed[0] = RandTurboPascal(masterSeed);
  for (unsigned i{ 1 }; i < numSeed; ++i)
    seed[i] = RandTurboPascal(seed[i - 1]);
  
  // **********************************
  unsigned* buffDev;
  SetUpDev(&buffDev, numSeed, threadPerBlock);

  // **********************************
  MappedFileT mappedFile("D:\\WorkSpace\\RandNum.data");
  const byte_t fileAddr = mappedFile();

  // **********************************
  cudaError_t status;
  for (unsigned seq{ 0 }; seq < numSeq; ++seq) {
    GenBuff <<<numBlock, threadPerBlock>>> (buff, numSeed);
    CheckError(cudaGetLastError(), "GenBuff failed to launch");

    // ToDo:  fix this so that it can be 64 bit
    constexpr unsigned buffSize = numSeed * sizeof(unsigned);
    const size_t filePos = (size_t)seq * buffSize;

    CheckError(cudaMemcpy(fileAddr[filePos], buffDev, buffSize), "Memory copy failed");
  }

  // **********************************
  CloseDown();

  return 0;
}
