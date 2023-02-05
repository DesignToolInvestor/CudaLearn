#include "UtilMiscCokie.h"

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stddef.h>

#include <format>
#include <iostream>
#include <random>


__global__ void GpuAddReduceKernel00(float* g_idata, float* g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;

    //covnert global data pointer to the local pointer
    // of this block
    float* block = g_idata + blockIdx.x * blockDim.x;

    // boundary check
    if (tid >= n) return;

    // in-place reduction in global memory
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (((tid % (2 * stride)) == 0) && ((tid + stride) < blockDim.x)) {
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