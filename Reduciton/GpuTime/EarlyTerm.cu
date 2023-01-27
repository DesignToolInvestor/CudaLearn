/*
  E a r l y T e r m . c u
*/

#pragma once

//#include <cuda.h>
#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//#include "GridHelper.cuh"

// ************************************
// This function assumes that the gird is 1D.
template<typename ElemT>
__global__ void AddReduceEarlyTerm(ElemT* partSum, ElemT* data, unsigned dataSize)
{
  // ToDo:  compare speeds with using size_t instead of unsigned
  unsigned numBlock = gridDim.x;
  unsigned threadPerBlock = blockDim.x;
  unsigned elemPerBlock = threadPerBlock * 2;

  unsigned blockNum = blockIdx.x;
  unsigned blockThread0 = blockNum * threadPerBlock;
  //unsigned blockElem0 = blockThread0 * 2;

  unsigned localThread = threadIdx.x;
  //unsigned globalThread = localThread + blockThread0;

  unsigned globalElem = 2 * blockThread0 + localThread;

  // This is the portion of the block/array that is active ... shrinks with each iteration.
  unsigned numActiveElem, numActiveThread;

  if (blockNum < numBlock - 1) {
    numActiveElem = elemPerBlock;
    numActiveThread = threadPerBlock;
  }
  else {
    numActiveElem = dataSize - 2 * blockThread0;
    numActiveThread = (numActiveElem + 1) >> 1;
  }

  // Do this thread's computation
  unsigned localCompanionElem = localThread + numActiveThread;
  if (localCompanionElem < numActiveElem) {
    unsigned otherGlobalElem = globalElem + numActiveThread;
    data[globalElem] += data[otherGlobalElem];

    __syncthreads();

    // Higher numbered threads will finish early
    unsigned count = 0;
    while (((localThread < numActiveThread) && (1 < numActiveThread)) && (count < 1)) {
      unsigned numActiveElem = numActiveThread;
      numActiveThread = (numActiveElem + 1) >> 1;

      localCompanionElem = localThread + numActiveThread;
      if (localCompanionElem < numActiveElem) {
        unsigned otherGlobalElem = globalElem + numActiveThread;
        data[globalElem] += data[otherGlobalElem];
      }
      __syncthreads();
    }
  }

  // copy partSum back
  if (localThread == 0)
    partSum[blockNum] = data[globalElem];
}

// Do instantiations
template __global__ void AddReduceEarlyTerm(float* partSum, float* data, unsigned dataSize);