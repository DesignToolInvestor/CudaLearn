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
#include "RandSeq.h"
#include "SeedManagement.h"
#include "UtilMisc.h"

// Project
#include "../Library/ReduceAdd.h"
#include "EarlyTerm.h"

using namespace std;

// ************************************
typedef float ElemT;
typedef double CheckT;

TickCountT ReadTicks_d()
{
  return ReadTicks();
}

float TicksToSecs_d(TickCountT ticks)
{
  return TicksToSecs(ticks);
}

// ************************************
int main()
{
  constexpr size_t minSize = 10;
  constexpr float approxMaxSize = 200e6;
  constexpr unsigned sampPerDec = 8;

  const double stepFact = exp(log(10) / sampPerDec);
  const unsigned numIter = (unsigned)round(log(approxMaxSize/minSize) / log(stepFact)) + 1;

  constexpr unsigned threadPerBlock = 256;

  // Setup random number generator
  const int seed = TimeSeed(6)();
  RandSeqFloat rand(0, 1, seed);

  // Do loop of increasing sizes
  double aimSize = minSize;
  for (unsigned iter{ 0 }; iter < numIter; iter++) {
    unsigned size = (unsigned)round(aimSize);
    vector<float> data(size);
    for (float& elem : data)
      elem = rand();

    ElemT result;
    ReduceAddGpu<ElemT>(result, &data[0], size, threadPerBlock);

    // Check accuracy
    assert(size < INT_MAX);
    CheckT answer = ReduceAdd<ElemT, CheckT>(&data[0], (unsigned)size);
    ElemT relError = (ElemT)(((CheckT)result - answer) / answer);
    
    // Print Result
    //cout << size <<  ", " << relError << '\n';

    aimSize = stepFact * aimSize;
  };

  return 0;
}