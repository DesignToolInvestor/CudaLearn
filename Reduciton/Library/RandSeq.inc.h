/*
  R a n d S e q . i n c . h
*/

#pragma once

//#include <chrono>
#include "SeedManagement.h"

// Included for the IDE
#include "RandSeq.h"

// ************************************
// Static allocation
TimeSeed randSeqTimsSeed(6);

// ************************************
// RandSeqInt
RandSeqInt::RandSeqInt(int low, int high) : 
  seed(randSeqTimsSeed()), coreGen(seed), dist(low, high)
    { /* Nothing */ };

RandSeqInt::RandSeqInt(int low, int high, int seedArg) :
  seed(seedArg), coreGen(seedArg), dist(low, high)
{ /* Nothing */
};

int RandSeqInt::operator() ()
{
  return dist(coreGen);
}

int RandSeqInt::Seed()
{
  return seed;
}

// ************************************
// RandSeqFloat
RandSeqFloat::RandSeqFloat(float low, float high) :
  seed(randSeqTimsSeed()), coreGen(seed), dist(low, high)
{ /* Nothing */
};

RandSeqFloat::RandSeqFloat(float low, float high, int seedArg) :
  seed(seedArg), coreGen(seed), dist(low, high)
{ /* Nothing */
};

float RandSeqFloat::operator() ()
{
  return dist(coreGen);
}

int RandSeqFloat::Seed()
{
  return seed;
}