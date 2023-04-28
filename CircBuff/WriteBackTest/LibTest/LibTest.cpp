// LibTest.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>

#include "OldRand.h"

using namespace std;

int main()
{
  constexpr unsigned numRand = 10;
  constexpr unsigned numPerLine = 5;

  unsigned seed = 0;

  for (unsigned count{ 0 }; count < (numRand - 1); ++count) {
    cout << seed;

    if ((count % numPerLine) == (numPerLine - 1))
      cout << '\n';
    else
      cout << ", ";

    seed = RandTurboPascal(seed);
  }

  cout << seed << '\n';

  return 0;
}