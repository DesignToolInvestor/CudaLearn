/*
  D e v I n f o . c u h
*/

#pragma once

class DevInfo {
protected:
  unsigned numDev, numSm, numF32CorePerSm;

protected:
  void CheckOk(const cudaError_t status);

public:
  DevInfo();

  unsigned NumDev() const;
  unsigned NumF32CorePerSm() const;
};