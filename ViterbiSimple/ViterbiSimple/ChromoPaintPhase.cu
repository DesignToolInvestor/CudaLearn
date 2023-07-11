/*
  C h r o m o P a i n t P h a s e . c u
*/

// Modern C++
#include <cstdlib>
#include <iostream>

// Cuda
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// *************************************
// ToDo:  Make this a template
typedef struct {
  float logLike;
  uint32_t prev;
} NodeT;

// *************************************
template<typename ProbT>
inline ProbT LogReadProb(
  unsigned personIdx, unsigned snpIdx, unsigned hapIdx, ProbT read[], ProbT logReadProb[])
{
  return logReadProb[read[personIdx, snpIdx], read[personIdx, hapIdx]]];
}

// *************************************
template<typename ProbT, typename stateIndexT>
__global__ void ChromoPaintGroupPhaseForKern(
  NodeT lattice[],
  const unsigned numBuff, const unsigned numPeople, const unsigned numSnp,
  const uint8_t read[], 
  const ProbT logProbNoTrans[], const ProbT logProbTrans[], const ProbT logReadProb[],
  unsigned toDoStart, unsigned toDoNum)
{
  // ToDo:  This doesn't fit.  Need to rethink everything.
  __shared__ buff[numBuff, numPeople, numState];

  // Unsigned thread = threadIdx.x;
  unsigned idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  unsigned personIdx = idx / numPeople;
  unsigned stateIdx = idx % numPeople;

  // Load first column
  latus[0, personIdx, stateIdex] = buff[0, personIdx, stateIdx] = logHapWeight[stateIdx];
  __syncthreads();

  // Do the rest of the columns
  for (unsigned snpIdx = 1; snpIdx < numSnp; ++snpIdx) {
    unsigned buffIdx = snpIdx % numBuff;
    unsigned prevBuffIdx = (snpIdx - 1) % numBuff;

    // Find max of previous column for each read

    // Find the best transition for my read
    ProbT noTransLogLike = 
      buff[prevBuffIdx, personIdx, stateIdx] + logProbNoTrans[stateIdx] + 
      LogReadProb<>(personIdx, snpIdx, stateIdx, read, logReadProb);

    ProbT bestTransLogLike =
      buff[prevBuffIdx, personIdx, stateIdx] + logProbNoTrans[stateIdx] +
      LogReadProb<>(personIdx, snpIdx, stateIdx, read, logReadProb);

    if (noTransLogLike < bestTransLogLike) {
      bestPrev = stateIdx;
    } else {

    }

    // Write answer out
    buff[buffIdx, personIdx, stateIdx].logLike = bestLogLike;
    buff[buffIdx, personIdx, stateIdx].prev = bestPrev;
  }
}