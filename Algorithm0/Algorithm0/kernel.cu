
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

//Todo: make this a template function
__device__
float* index2D(float* arr, int num_rows, int num_cols, int row_indx, int col_indx) {
    int index = (num_cols * row_indx) + col_indx;
    return &arr[index];
}

__device__
float max(float a, float b) {
    if (a < b) {
        return b;
    }
    else return a;
}

// prob_matrix is output
// haps is every haplotype, where each row is a haplotype
__global__ void lsf_kernel(float *prob_matrix, float *haps, float *gmap,
    const int target, const int num_snp, const int num_hap, const float read_error)
{
    __shared__ float buff0[num_hap];
    __shared__ float buff1[num_hap];

    float* currColumn = buff0;
    float* prevColumn = buff1;

    float probRead = 0;
    float probTrans = 0;
    float outLogLike = 0;
    int thread = threadIdx.x;

    // load first buffer
    prevColumn[thread] = 1 / num_hap;
    __syncthreads();

    // Do each column
    for (int snp_num = 1; snp_num < num_snp; snp_num++) {
        // calculate log likelihoods
        for (int hap_num = 0; hap_num < num_hap; hap_num++) {
            // don't compare target to itself, or it will always match itself
            if (hap_num != target) {
                // get emission probability
                if (*index2D(haps, num_hap, num_snp, target, snp_num)
                    == *index2D(haps, num_hap, num_snp, hap_num, snp_num))
                    probRead = 1 - read_error;
                else
                    probRead = read_error;

                // get transition probability
                if (target == hap_num)
                    probTrans = gmap[snp_num] - gmap[snp_num - 1];
                else
                    probTrans = 1 - (gmap[snp_num] - gmap[snp_num - 1]);

                // update current max probability explanation for observation
                outLogLike =
                    max(outLogLike, prevColumn[hap_num] + probTrans + probRead);
            }

            currColumn[thread] = outLogLike;
            __syncthreads();

            // copy curr column from shared memory out to global probability matrix
            *index2D(prob_matrix, num_hap, num_snp, snp_num, thread) = currColumn[thread];
            // swap prevColumn with currColumn in our circular buffer
            float* temp = prevColumn;
            prevColumn = currColumn;
            currColumn = temp;

            __syncthreads();
        }

        // sync threads
        __syncthreads();
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
