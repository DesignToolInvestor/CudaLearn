#pragma once

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stddef.h>

#include <format>
#include <iostream>
#include <random>

#include "../Library/ReduceAdd.h"

template<typename ElemT>
cudaError_t ReduceAddGpu(const ElemT* data, int dataSize, ElemT& result);