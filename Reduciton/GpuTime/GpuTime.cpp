/*
  G p u T i m e . c p p
*/

// Modern C++
#include <cmath>
#include <format>
#include <iostream>
#include <random>
#include <vector>

// Legacy C
#include <stdio.h>
#include <stddef.h>

// Cuda
#include <cuda.h>
#include "cuda_runtime.h"

// Kenneth library
#include "SeedManagement.h"
#include "RandSeq.h"

// Project
#include "../Library/ReduceAdd.h"
#include "EarlyTerm.h"

using namespace std;

// ************************************
typedef float ElemT;
typedef double CheckT;

int main()
{
  constexpr size_t minSize = 10;
  constexpr float maxSize = 1.05e9;
  constexpr unsigned sampPerDec = 8;
  constexpr float stepSize = (float)exp(log(10) / sampPerDec);

  constexpr unsigned threadPerBlock = 256;

  // Setup random number generator
  const int seed = TimeSeed(6)();
  RandSeqFloat rand(0, 1, seed);

  // Do loop of increasing sizes
  size_t size = minSize;
  do {
    vector<float> data(size);
    for (float& elem : data)
      elem = rand();

    ElemT result;
    ReduceAddGpu<ElemT>(result, &data[0], size, threadPerBlock);

    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "cudaDeviceReset failed!");
      return 1;
    }

    // Check accuracy
    assert(size < INT_MAX);
    CheckT answer = ReduceAdd<ElemT, CheckT>(&data[0], (unsigned)size);
    ElemT relError = (ElemT)(((CheckT)result - answer) / answer);
    
    // Print Result
    cout << size <<  ", " << relError << '\n';

    size = (size_t)round(step * size);
  } while (size <= maxSize);

  return 0;
}