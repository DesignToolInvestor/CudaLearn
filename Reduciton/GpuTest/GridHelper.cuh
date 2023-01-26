/*
  G r i d H e l p e r . c u h
*/

#pragma once

#include <cmath>

#include <cuda.h>
#include "cuda_runtime.h"


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