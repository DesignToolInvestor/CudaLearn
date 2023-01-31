/*
  G p u T e s t . c p p
*/

// Modern C++
#include <cmath>
#include <format>
#include <iostream>
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

// Project
#include "../Library/ReduceAdd.h"
//#include "EarlyTerm.cuh"

// ************************************
// ToDo: replace with include of EarlyTerm.h
template<typename ElemT>
  void ReduceAddGpu(
    ElemT& result, const ElemT* data, size_t numElem, unsigned threadPerBlock);

// ************************************
int main()
{
  constexpr size_t minSize = 10;
  constexpr size_t maxSize = 64'000;
  const float step = (float)exp(log(10)/4);
  constexpr unsigned threadPerBlock = 256;

  size_t size = minSize;
  while (size <= maxSize) {
    int* data = new int[size];

    assert(size < INT_MAX);
    for (size_t i = 0; i < size; i++)
      data[i] = (int)i;

    int result;
    ReduceAddGpu<int>(result, data, size, threadPerBlock);
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

    size = (size_t)round(step * size);
  }

  return 0;
}