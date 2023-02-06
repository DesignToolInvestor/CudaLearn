#include "UtilMiscCokie.h"

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stddef.h>

#include <format>
#include <iostream>
#include <random>


__global__ void GpuAddReduceKernel01(float* g_idata, float* g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;

    //covnert global data pointer to the local pointer
    // of this block
    float* block = g_idata + blockIdx.x * blockDim.x;

    // boundary check
    if (tid >= n) return;

    unsigned int start_stride = ((n - 1) / 2) + 1;
    // in-place reduction in global memory
    for (int stride = start_stride; stride >= 1; stride = ((stride - 1) / 2) + 1) {
        if (( ((tid + stride) < blockDim.x) && ((tid + stride) < n) )) {
            block[tid] += block[tid + stride];
        }

        //debug
        /*if (tid == 0) {
            printf("stride: %d , blockSize: %d\n", stride, blockDim.x);
        }*/

        // synchronize within block
        __syncthreads();
    }

    //write result for this block to global mem
    if (tid == 0) {
        g_odata[blockIdx.x] = block[0];

        //debug
        /*printf("thread %d result: %f \n", tid, *block);*/
    }
}