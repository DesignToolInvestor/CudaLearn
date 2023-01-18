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
    maxBlockPerSm, maxThreadPerSm;

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

  unsigned MaxBlockPerSm() const;
  unsigned MaxThreadPerSm() const;
};