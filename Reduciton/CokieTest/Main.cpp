
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stddef.h>

#include <format>
#include <iostream>
#include <random>

#include "../Library/ReduceAdd.h"
#include "UtilMiscCokie.h"
#include "GpuAddReduce.h"


using namespace std;

int trueAnswer(int numElems) {
    return (numElems * (numElems - 1)) / 2;
}

// ************************************
int main()
{
    const int blockSize = 512;

    constexpr unsigned startN = 1'000'000;
    constexpr unsigned stepPerDec = 2;
    constexpr unsigned stopN = 100'000'000;

    const double nStepFact = (float)exp(log(10) / stepPerDec);
    //const double nStepFact = 1;

    //const int seed = TimeSeed(6)();
    //RandSeqFloat rand(0, 1, seed);

    // create array to be add-reduced
    for (int dataSize = startN; dataSize <= stopN; dataSize *= nStepFact) {
        float* data = new float[dataSize];
        for (int i = 0; i < dataSize; i++) {
            data[i] = i;
        }

        // debug
        /*for (int i = 0; i < dataSize; i++) {
            std::cout << data[i] << " ";
        }
        std::cout << "\n";*/

        // add-reduce array
        float result = -1;
        cudaError_t status = ReduceAddGpu<float>(result, data, dataSize, blockSize);

        // Need a delete [] data
        delete[] data;

        //debug
        std::cout << "data size: " << dataSize << "\n";
        std::cout << "result: " << result << "\n";
        float trueAns = trueAnswer(dataSize);
        float error = (trueAns - result) / trueAns;
        std::cout << "true answer: " << trueAns << " error: " << error << "\n\n";
    }
}