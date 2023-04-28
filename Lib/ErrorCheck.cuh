/*
  E r r o r C h e c k . c u h
*/

#pragma once

// Modern C++
#include <cstdlib>
#include <iostream>

// Cuda
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// ************************************
void CheckErr(cudaError_t status, const char* message)
{
  if (status != cudaSuccess) {
    std::cout << message;
    abort();
  }
}

// ************************************
void CheckErr(cudaError_t status)
{
  if (status != cudaSuccess) {
    std::cout << cudaGetErrorString(status) << '\n';
    abort();
  }
}