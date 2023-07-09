/*
  G p u T e s t . c p p
*/

// Modern C++
#include <cmath>
#include <format>
#include <iostream>
//#include <limits>
#include <random>

// Legacy C
#include <stdio.h>
#include <stddef.h>

// Cuda
#include <cuda.h>
#include "cuda_runtime.h"

// Kenneth library
#include "SeedManagement.h"
#include "RandSeq.h"

// Solution Library
#include "../Library/ReduceAdd.h"

// Local to project
#include "ReduceAddWrap.h"

using namespace std;

// ************************************
// ToDo:  These are called by ReduceAdd_d, not a good system
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
  // Setup
  constexpr size_t minSize = 16;

  // This is the minimum array size at which overflow occurs.
  // Quadratic formula
  const float maxSize = (1 + sqrt(1 + 8 * (float)INT_MAX)) / 2;

  constexpr unsigned stepPerDouble = 2;
  const float stepFact = (float)exp(log(2) / stepPerDouble);
  
  constexpr unsigned threadPerBlock = 512;

  // Loop
  float aimSize = minSize;
  size_t size = minSize;

  while (aimSize <= maxSize) {
    int* data = new int[size];

    assert(size < INT_MAX);
    for (size_t i = 0; i < size; i++)
      data[i] = (int)i;

    int result;
    ReduceAddWrap<int>(result, data, size, threadPerBlock);
    delete[] data;

    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "cudaDeviceReset failed!");
      return 1;
    }

    if (result != (size - 1) * size / 2) {
      fprintf(stderr, "Got wrong answer!");
      return 1;
    }

    size = (size_t)round(stepFact * size);
  }

  return 0;
}