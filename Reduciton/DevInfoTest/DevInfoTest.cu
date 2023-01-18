/*
  D e v I n f o T e s t . c u
*/

#include <iostream>
#include <format>

#include "DevInfo.cuh"

using namespace std;

// ************************************************************

int main()
{
  DevInfo devInfo;

  unsigned numDev = devInfo.NumDev();
  if (numDev == 0)
    cout << "Doesn't have a Cuba comparable device\n";
  else if (1 < numDev)
    cout << "Haven't implemented code for multiple Cuba devices\n";
  else {
    cout << devInfo.NumDev() << " GPUs\n";
    cout << devInfo.NumSm() << " SMs\n\n";

    cout << "  Computation class = " << devInfo.CompClassMajor() << '.' << 
      devInfo.CompClassMinor() << '\n';
    cout << "  F32 cores per SM = " << devInfo.NumF32CorePerSm() << '\n';

    cout << "  Max blocks per SM = " << devInfo.MaxBlockPerSm() << '\n';
    cout << "  Max thread per SM = " << devInfo.MaxThreadPerSm() << '\n';
  }
}