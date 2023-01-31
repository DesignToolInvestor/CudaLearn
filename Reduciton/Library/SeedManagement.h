/*
  S e e d M a n a g e m e n t . h
*/

#pragma once

#include "UtilMisc.h"

static uint64_t BitRev(uint64_t inVal, int numBit);

class TimeSeed {
protected:
  int numDig;
  static uint64_t lastTimeUsed;

protected:
  static uint64_t EpocTime();

public:
  TimeSeed(int numDig);
  int operator() ();
};

class SeedGroup {
protected:
  int numDig, numBit;
  uint64_t baseSeed;
  const double scale;

public:
  SeedGroup(uint64_t baseSeed, int numDig);
  uint64_t operator[] (int seedNum);
};

// No separate compilation at this point
#include "SeedManagement.inc.h"