﻿
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);


// prob_matrix is output
// haps is every haplotype, where each row is a haplotype
__global__ void lsf_kernel(float *prob_matrix, float *haps, float *gmap, const int target, const int num_snp, const int num_hap, const float read_error)
{
    __shared__ float buff0[num_hap];
    __shared__ float buff1[num_hap];

    float* currColumn = buff0;
    float* prevColumn = buff1

    float probRead = 0;
    int thread = threadIdx.x;

    // load first buffer
    prevColumn[thread] = 1 / num_hap;
    __syncthreads();

    // Do each column
    for (int snp_num = 1; snp_num < num_snp; snp_num++)
        // calculate log likelihoods
        for (int hap_num = 0; hap_num < num_hap; hap_num++) {
            if (hap_num != target) {
                if (haps[target][snp_num] == haps[hap_num][snp_num])
                    probRead = 1 - read_error;
                else
                    probRead = read_error;

                if (target == hap_num)
                    logProbTrans = logOneMinusExpMorgan[i];
                else
                    logProbTrans = morgan[i] + weight[j];

                ourLogLike =
                    max(ourLogLike, prevColumn[hap_num] + logProbTrans + logProbRead);

                //write to curr_prob
                curr_prob[] = logLike[k, i];
            }

            currColumn[thread] = outLogLike;
            __syncthread();

            prob_matrix[snp_num][thread] = currColumn[thread];
            Swap(prevColumn, currColun);

            currColumn[thread] = prob_matrix[][];
            __syncthread();
        }

        // sync threads
        __syncthreads
    }
}

int main()
{
    // define data

    // allocate memory

    // copy memory

    // call kernel

    // copy memory back

    // print answer and matrix

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
