/*
  C h r o m o P a i n t U n P h a s e . c u
*/

#pragma once

// Modern C++
#include <cstdlib>
#include <iostream>

// Cuda
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// ************************************
template<typename ProbT, typename StateIdxT>
class NodeT {
public:
  ProbT logLike;
  StateIdxT prev;
};

// ************************************************************
template<typename ProbT, typename StateIdxT>
__global__ void ChromoPaintUnPhaseForKern(
  NodeT<ProbT, StateIdxT> lattice[],
  const unsigned numBuff, const unsigned numPeople, const unsigned numSnp,
  const unsigned firstPerson, const unsigned lastPerson,
  const uint8_t read[],
  const ProbT logProbNoTrans[], const ProbT logProbTrans[], const ProbT logReadProb[],
  unsigned toDoStart, unsigned toDoNum);

// ************************************
template<typename ProbT, typename StateIdxT>
__global__ void ChromoPaintUnPhaseRevKern(

  const unsigned numBuff, const unsigned numPeople, const unsigned numSnp,
  const NodeT<ProbT, StateIdxT> lattice[],
  const ProbT logProbNoTrans[], const ProbT logProbTrans[], const ProbT logReadProb[],
  unsigned toDoStart, unsigned toDoNum);