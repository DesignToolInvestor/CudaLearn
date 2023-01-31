/*
  S e e d M a n a g e m e n t . h
*/

#pragma once

#include <stdexcept>
#include <chrono>

// included to aid the IDE
#include "SeedManagement.h"

// ************************************
// Bit reversal function
uint64_t BitRev(uint64_t inVal, int numBit)
{
  uint64_t outVal = 0;

  for (int i = 0; i < numBit; i++) {
    outVal = (outVal << 1);
    outVal = outVal | (inVal & 1);
    inVal = (inVal >> 1);
  }

  if (inVal != 0)
    throw std::invalid_argument("Input value too larger for the number of bits");

  return outVal;
}

// ************************************
// TimeSeed
TimeSeed::TimeSeed(int numDig) : numDig(numDig)
{
  if (20 < numDig)
    throw std::invalid_argument("Maximum of 20 digits");
};

uint64_t TimeSeed::EpocTime()
{
  return std::chrono::system_clock::now().time_since_epoch().count();
}

int TimeSeed::operator() ()
{
  // ToDo:  Make this tread safe.
  uint64_t now = EpocTime();
  uint64_t timeToUse = Max(lastTimeUsed + 1, now);
  lastTimeUsed = timeToUse;
  // End of critical section of code.

  uint64_t bitRevTime = BitRev(timeToUse, 64);
  int seedToUse = (int)floor(pow(10, numDig) * double(bitRevTime) / pow(2, 64));
  return seedToUse;
}

// ************************************
// SeedGroup
SeedGroup::SeedGroup(uint64_t baseSeed, int numDig) :
  numDig(numDig), numBit((int) ceil(numDig* log(10) / log(2))),
  baseSeed(baseSeed),
  scale(pow(10, numDig) / pow(2, numBit))
{
  if (20 < numDig)
    throw std::invalid_argument("Maximum of 20 digits");
}

// ToDo:  Rewrite so that we multiply the baseSeed by bit reversal of (seedNum + 1)
uint64_t SeedGroup::operator[] (int seedNum)
{
  uint64_t mappedBaseSeed = (uint64_t) round(baseSeed / scale);
  uint64_t revMappedBaseSeed = BitRev(mappedBaseSeed, numBit);

  uint64_t mappedOutSeed = BitRev(revMappedBaseSeed + seedNum, numBit);
  uint64_t nextSeed = (uint64_t) round(scale * mappedOutSeed);

  return nextSeed;
}

uint64_t TimeSeed::lastTimeUsed = 0;