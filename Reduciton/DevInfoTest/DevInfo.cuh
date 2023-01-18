/*
  D e v I n f o . c u h
*/

#pragma once

#include <string>
#include <cuda.h>

class DevInfo {
protected:
  std::string name;
  unsigned numDev, numSm, compClassMajor, compClassMinor, numF32CorePerSm,
    maxThreadPerBlock, maxThreadPerSm, maxBlockPerSm;

protected:
  void CheckOk(const cudaError_t status);

public:
  DevInfo();

  std::string Name() const;

  unsigned NumDev() const;
  unsigned NumSm() const;

  unsigned CompClassMajor() const;
  unsigned CompClassMinor() const;
  unsigned NumF32CorePerSm() const;

  unsigned MaxThreadPerBlock() const;
  unsigned MaxThreadPerSm() const;
  unsigned MaxBlockPerSm() const;
};