#include "UtilMiscCokie.h"

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stddef.h>

#include <format>
#include <iostream>
#include <random>


__global__ void GpuAddReduceKernel01(float* g_idata, float* g_odata, unsigned int arrSize)
{
    // set thread ID
    unsigned int tid = threadIdx.x;

    //covnert global data pointer to the local pointer
    // of this block
    float* block = g_idata + blockIdx.x * blockDim.x;

    // boundary check
    if (tid >= arrSize) return;

    unsigned int activeDataSize = arrSize;
    unsigned int startStride = ((arrSize - 1) / 2) + 1;
    // in-place reduction in global memory
    int loopCount = 0;
    for (int stride = startStride; (activeDataSize > 1) && (loopCount < 100); stride = ((stride - 1) / 2) + 1) {
        int companionThread = tid + stride;
        if ( (companionThread < blockDim.x) && (companionThread < activeDataSize) ) {
            block[tid] += block[companionThread];
        }

        //debug
        /*if (tid == 0) {
            printf("stride: %d , blockSize: %d\n", stride, blockDim.x);
        }*/
        

        // synchronize within block
        __syncthreads();

        activeDataSize = stride;

        loopCount++;
    }

    //write result for this block to global mem
    if (tid == 0) {
        g_odata[blockIdx.x] = block[0];

        //debug
        /*printf("thread %d result: %f \n", tid, *block);*/
    }
}