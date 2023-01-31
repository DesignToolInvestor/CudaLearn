/*
  C p u S i n g l e . c p p
*/

#include <iostream>
#include <cmath>

#include "../Library/SeedManagement.h"
#include "../Library/RandSeq.h"

#include "../Library/ReduceAdd.h"

using namespace std;

// ************************************
int main()
{
  constexpr unsigned startN = 100;
  constexpr unsigned stepPerDec = 8;

  const double nStepFact = exp(log(10)/stepPerDec);

  const int seed = TimeSeed(6)();
  RandSeqFloat rand(0,1, seed);

  float deltaSec;
  unsigned n = startN;
  do {
    // build data
    vector<float> data(n);
    for (float& elem : data)
      elem = rand();

    // Time reduction
    TickCountT start = ReadTicks();
    float result = ReduceAdd<float>(&data[0], data.size());
    deltaSec = TicksToSecs(ReadTicks() - start);

    // Check accuracy
    double answer = ReduceAdd<float,double>(&data[0], data.size());
    float relError = (float)(((double)result - answer) / answer);

    // Print Result
    cout << n << ", " << deltaSec << ", " << relError << '\n';

    n = max((unsigned)round(n * nStepFact), n + 1);
  } while (deltaSec < 1);

  std::cout << "Hello World!\n";
}