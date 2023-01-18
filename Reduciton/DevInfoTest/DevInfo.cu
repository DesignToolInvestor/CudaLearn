/*
  D e v I n f o . c u
*/

#pragma once

#include <cstdio>
#include "cuda_runtime.h"
#include "devInfo.cuh"

// ****************************************************************************
// ToDo:  Move to a separate file at some point

void DevInfo::CheckOk(const cudaError_t status)
{
  // ToDo:  Want to halt the debugger, but need to decide on an exception pattern
  if (status != CUDA_SUCCESS)
    abort();
}

// ************************************************************
DevInfo::DevInfo() {
  constexpr unsigned devNum = 0;

  // Initialize
  CheckOk(cudaSetDevice(devNum));
  
  // Get number of GPUs
  int temp;
  CheckOk(cudaGetDeviceCount(&temp));
  if (temp < 0)
    abort();
  numDev = (unsigned)temp;

  // Get device properties
  cudaDeviceProp devProp;
  CheckOk(cudaGetDeviceProperties(&devProp, 0));

  // Get stuff in devProp
  numSm = devProp.multiProcessorCount;
  compClassMajor = devProp.major;
  compClassMinor = devProp.minor;

  maxThreadPerBlock = devProp.maxThreadsPerBlock;
  maxThreadPerSm = devProp.maxThreadsPerMultiProcessor;
  maxBlockPerSm = devProp.maxBlocksPerMultiProcessor;

  // F32 cores per SM
  typedef struct {
    int major, minor, f32Cores;
  } CoreInfoT;

  constexpr unsigned numVer = 17;
  CoreInfoT smInfo[numVer] = {
    {3,0, 192}, {3,2, 192}, {3,5, 192}, {3,7, 192},
    {5,0, 128}, {5,2, 128}, {5,3, 128},
    {6,0,  64}, {6,1, 128}, {6,2, 128},
    {7,0,  64}, {7,2,  64}, {7,5,  64},
    {8,0,  64}, {8,6, 128}, {8,9, 128},
    {9,0, 128} };

  int i = 0;
  while ((i < numVer) && ((smInfo[i].major != compClassMajor) || (smInfo[i].minor != compClassMinor)))
    i++;

  if (i == numVer)
    abort();

  numF32CorePerSm = smInfo[i].f32Cores;
}

// ************************************************************
// Accessor functions
unsigned DevInfo::NumDev() const
{
  return numDev;
}

unsigned DevInfo::NumSm() const
{
  return numSm;
}

unsigned DevInfo::CompClassMajor() const
{
  return compClassMajor;
}

unsigned DevInfo::CompClassMinor() const
{
  return compClassMinor;
}

unsigned DevInfo::NumF32CorePerSm() const
{
  return numF32CorePerSm;
};

unsigned DevInfo::MaxThreadPerBlock() const
{
  return maxThreadPerBlock;
}

unsigned DevInfo::MaxThreadPerSm() const
{
  return maxThreadPerSm;
}

unsigned DevInfo::MaxBlockPerSm() const
{
  return maxBlockPerSm;
}
