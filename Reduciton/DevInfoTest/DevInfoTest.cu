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
    cout << "  Max thread per SM = " << devInfo.MaxThreadPerSm() << '\n\n';

    // ************************
    constexpr float saftyThread = 1.5;
    constexpr float saftyBlock = 1;
    constexpr unsigned avgLatency = 10;
    
    unsigned idealThreadPerSm = round(devInfo.NumF32CorePerSm() * avgLatency * saftyThread);
    unsigned threadPerSm = min(idealThreadPerSm, devInfo.MaxThreadPerSm());

    unsigned idealBlockPerSm = (unsigned)round(sqrt(2 * devInfo.MaxBlockPerSm()));
    unsigned idealThreadPerBlock = (unsigned)ceil(threadPerSm / idealBlockPerSm);

    unsigned theadPerBlock = min(idealBlockPerSm, devInfo.MaxThreadPerBlock());
    unsigned blockPerSm = (unsigned)ceil(threadPerSm / theadPerBlock);

    unsigned numBlock = blockPerSm * devInfo.NumSm() * saftyBlock;
    unsigned criticalNumThread = numBlock * theadPerBlock;

    cout << "Design heuristics (@ Safety = (1.5, 1), Latency = 10):\n";
    if (idealThreadPerSm = threadPerSm)
      cout << "  Threads per SM = " << idealThreadPerSm << '\n';
    else {
      cout << "  Ideal threads per SM = " << idealThreadPerSm << " exceeds max\n";
      cout << "    Actual = " << threadPerSm << '\n';
    }

    if (idealBlockPerSm = blockPerSm) {
      cout << "  Block per SM = " << blockPerSm << '\n';
      cout << "  Thread per block = " << theadPerBlock << '\n';
    } else {
      cout << "  Ideal block per SM = " << idealBlockPerSm << '\n';
      cout << "  Implied thread per block = " << idealThreadPerBlock << " exceeds maximum.\n";
      cout << "    Actual thread per block = " << theadPerBlock << '\n';
      cout << "    Actual block per SM = " << blockPerSm << '\n';
    }

    cout << "  Total number of block = " << numBlock << '\n';
    cout << "  Critical number of thread = " << criticalNumThread << '\n';
  }
}