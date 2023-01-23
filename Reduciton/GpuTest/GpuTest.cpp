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
cudaError_t ReduceAddGpu(
  ElemT& result, const ElemT* data, size_t numElem, unsigned threadPerBlock);

// ************************************
int main()
{
  constexpr size_t minSize = 15;
  constexpr size_t maxSize = 17;
  constexpr unsigned threadPerBlock = 8;

  for (size_t size{ minSize }; size <= maxSize; size++) {
    int* data = new int[size];
    for (size_t i = 0; i < size; i++)
      data[i] = i;

    int result;
    cudaError_t cudaStatus = ReduceAddGpu<int>(result, data, size, threadPerBlock);
    delete[] data;

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "cudaDeviceReset failed!");
      return 1;
    }

    if (result != (size - 1) * size / 2) {
      fprintf(stderr, "Got wrong answer!");
      return 1;
    }
    else
      fprintf(stderr, "Size = %d passed\n", size);
  }

  return 0;
}