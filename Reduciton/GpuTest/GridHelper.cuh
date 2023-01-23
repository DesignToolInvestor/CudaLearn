/*
  G r i d H e l p e r . c u h
*/

#pragma once

#include <cmath>

#include <cuda.h>
#include "cuda_runtime.h"

// ************************************
dim3 GridSizeSimple(size_t activeThread, unsigned threadPerBlock, cudaDeviceProp devProp)
{
  // Compute gird parameters
  const unsigned warpSize = 32;
  const unsigned numWarp = (activeThread - 1) / warpSize + 1;
  const unsigned allThread = warpSize * numWarp;

  const unsigned numBlock = (allThread - 1) / threadPerBlock + 1;

  unsigned xSize = 1;
  unsigned ySize = 1;
  unsigned zSize = 1;

  if (numBlock < devProp.maxGridSize[0])
    xSize = numBlock;
  else if (numBlock < devProp.maxGridSize[1] * devProp.maxGridSize[1]) {
    xSize = warpSize * (unsigned)ceil(sqrt(numBlock) / warpSize);
    ySize = (numBlock - 1) / xSize + 1;
  }
  else {
    xSize = devProp.maxGridSize[0];
    ySize = devProp.maxGridSize[1];
    zSize = (allThread - 1) / xSize / ySize + 1;
  }

  dim3 result{ xSize,ySize,zSize };
  return result;
}

// ************************************
// ToDo:  create template for less than 3 dimensions
__device__ unsigned BlockSize()
{
  return blockDim.x * blockDim.y * blockDim.z;
}

__device__ unsigned GridSize()
{
  return gridDim.x * gridDim.y * gridDim.z;
}

__device__ unsigned BlockNum()
{
  return (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
}

__device__ unsigned LocalThread()
{

  return (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
}

__device__ unsigned BlockToGlobalThread()
{
  return BlockNum() + BlockSize();
}

__device__ unsigned GlobalThread()
{
  return BlockToGlobalThread() + LocalThread();
}