/*
  G p u T e s t . c p p
*/

#include <format>
#include <iostream>
#include <random>

#include <stdio.h>
#include <stddef.h>

#include <cuda.h>
#include "cuda_runtime.h"
//#include "device_launch_parameters.h"

// ToDo:  not picking up environment variable
//#define KENNETH_LIB_DIR F:\Users\Kenne.DESKTOP-BT6VROU\Documents\GitHub\MANet-Sim\Lib\KennethLib\KennethLib
//#include "KENNETH_LIB_DIR\SeedManagement.h"

// Kenneth library
#include "SeedManagement.h"
#include "RandSeq.h"

#include "../Library/ReduceAdd.h"
//#include "EarlyTerm.cuh"

// ************************************
template<typename ElemT>
  void ReduceAddGpu(
    ElemT& result, const ElemT* data, size_t numElem, unsigned threadPerBlock);

// ************************************
int main()
{
  constexpr size_t minSize = 10;
  constexpr size_t maxSize = 64'000;
  constexpr float step = 1.3;
  constexpr unsigned threadPerBlock = 256;

  size_t size = minSize;
  while (size <= maxSize) {
    int* data = new int[size];
    for (size_t i = 0; i < size; i++)
      data[i] = i;

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