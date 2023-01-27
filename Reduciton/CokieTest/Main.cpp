
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
    constexpr unsigned startN = 100;
    constexpr unsigned stepPerDec = 8;

    const double nStepFact = exp(log(10) / stepPerDec);

    //const int seed = TimeSeed(6)();
    //RandSeqFloat rand(0, 1, seed);

    // create array to be add-reduced
    for (int dataSize = 6; dataSize < 7; dataSize++) {
        float* data = new float[dataSize];
        for (int i = 0; i < dataSize; i++) {
            data[i] = i;
        }

        // debug
        for (int i = 0; i < dataSize; i++) {
            std::cout << data[i] << " ";
        }
        std::cout << "\n";

        // add-reduce array
        float result = -1;
        cudaError_t status = ReduceAddGpu<float>(data, dataSize, result);

        // Need a delete [] data
        delete[] data;

        std::cout << "result: " << result << "\n\n";
        if (result != trueAnswer(dataSize)) {
            std::cout << "False!\n\n";
            abort();
        }
    }
}