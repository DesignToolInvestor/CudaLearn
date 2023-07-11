/*
  C h r o m o P a i n t U n P h a s e . c u
*/

#include "ChromoPaintUnPhase.cuh"

// ************************************
template<typename ProbT>
inline ProbT LogReadProb(
  unsigned readVal, unsigned hapVal, ProbT logReadGood[], ProbT logReadErr[])
{
  ProbT result;
  if (readVal == hapVal)
    result = logReadGood[haVal];
  else
    result = logReadErr[havVal];

  return result;
}

// ************************************************************
// ToDo:  create name spaces for ChromoPaint and UnPhased
template<typename ProbT, typename StateIdxT>
__global__ void ChromoPaintUnPhaseForStepKern(
  NodeT<ProbT, StateIdxT> column[],
  const unsigned numBuff, const unsigned numHap, const unsigned numSnp,
  const ProbT prevColumn[], const uint8_t read, const uint8_t hapAtSnp[],
  const ProbT logProbNoTrans, const ProbT logProbTrans, 
  const ProbT logReadErr[], const ProbT logReadGood[])
{
  assert(blockDim.x == numHap);
  unsigned personIdx = blockIdx.x;
  unsigned currState = threadIdx.x;

  // Will fit in memory up to about 19.6K, but suboptimal above about 6K
  __shared__ ProbT prevColumnLoc[numHap];

  prevColumnLoc[stateIdx] = prevColumn[stateIdx];

  __syncthreads();
  // **************************

  ProbT bestLogLike =
    prevColumnLoc[hapIdx] + logProbNoTrans + LogReadProb(read, hapAtSnp[currState]);
  StateIdxT bestPrev = currState;

  for (unsigned hapIdx = 0; hapIdx < numHap; ++hapIdx)
    if (hapIdx != currState) {
      ProbT logLike = 
        prevColumnLoc[hapIdx] + logProbTrans + LogReadProb(read, hapAtSnp[hapIdx]);

      if (bestLogLike < logLike) {
        bestLogLike = logLike;
        bestPrev = hapIdx;
      }
    }
  
  // Likely latency trap.  Will probably need to change, but the cache might avoid the problem.
  column[currState].logLike = bestLogLike;
  column[currState].bestPrev = bestPrev;
}
