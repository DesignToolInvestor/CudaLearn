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
    cout << devInfo.NumF32CorePerSm() << '\n';
  }
}