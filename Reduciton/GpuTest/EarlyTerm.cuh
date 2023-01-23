/*
  E a r l y T e r m . c u h
*/

#pragma once

//#include <cuda.h>
#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
#include "GridHelper.cuh"

// ************************************
template<typename ElemT>
  __global__ void AddReduceEarlyTerm(ElemT* partSum, ElemT* data, unsigned elemPerArray)
{
  // ToDo:  compare speeds with using size_t instead of unsigned
  unsigned blockNum = BlockNum();
  unsigned numBlock = GridSize();

  unsigned threadPerBlock = BlockSize();
  unsigned elemPerBlock = threadPerBlock * 2;

  unsigned blockThread0 = BlockToGlobalThread();
  unsigned blockElem0 = blockThread0 * 2;

  unsigned localThread = LocalThread();
  unsigned globalThread = GlobalThread();
  unsigned glogalElem = 2 * globalThread;

  // This is the portion of the block/array that is active. Shrinks with each iteration.
  unsigned numActiveElem, numActiveThread;

  // The last block may not start out full
  if (blockNum < numBlock - 1) {
    numActiveElem = elemPerBlock;
    numActiveThread = threadPerBlock;
  } else {
    numActiveElem = elemPerArray - 2 * blockThread0;
    numActiveThread = (numActiveElem + 1) >> 1;
  }

  // Do this thread's computation
  unsigned otherLocalElem = localThread + numActiveThread;
  if (otherLocalElem < numActiveElem) {
    unsigned otherGlobalElem = blockElem0 + numActiveThread;
    data[glogalElem] += data[otherGlobalElem];

    __syncthreads();

    // Higher numbered threads will finish early
    while ((localThread < numActiveThread) && (1 < numActiveThread)) {
      unsigned numActiveElem = numActiveThread;
      numActiveThread = (numActiveElem + 1) >> 1;

      otherLocalElem = localThread + numActiveThread;
      if (otherLocalElem < numActiveElem) {
        unsigned otherGlobalElem = blockElem0 + numActiveThread;
        data[glogalElem] += data[otherGlobalElem];
      }
      __syncthreads();
    }
  }

  // copy partSum back
  if (localThread == 0)
    partSum[blockNum] = data[blockThread0];
}
