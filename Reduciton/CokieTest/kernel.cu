﻿
#include <iostream>
#include <cmath>

#include "SeedManagement.h"
#include "RandSeq.h"

#include "../Library/AddReduceSerial.h"

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addReduceKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main()
{
    constexpr unsigned startN = 100;
    constexpr unsigned stepPerDec = 8;

    const double nStepFact = exp(log(10) / stepPerDec);

    const int seed = TimeSeed(6)();
    RandSeqFloat rand(0, 1, seed);

    float deltaSec;
    unsigned n = startN;
    do {
        // build data
        vector<float> data(n);
        for (float& elem : data)
            elem = rand();

        // Time reduction
        TickCountT start = ReadTicks();
        float result = ReduceAddCpu<float>(data);
        deltaSec = TicksToSecs(ReadTicks() - start);

        // Check accuracy
        double answer = ReduceAddCpu<float, double>(data);
        float relError = (float)(((double)result - answer) / answer);

        // Print Result
        cout << n << ", " << deltaSec << ", " << relError << '\n';

        n = max((unsigned)round(n * nStepFact), n + 1);
    } while (deltaSec < 1);

    std::cout << "Hello World!\n";
}