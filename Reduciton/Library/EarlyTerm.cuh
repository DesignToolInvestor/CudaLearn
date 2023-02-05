/*
  E a r l y T e r m . c u h
*/

#pragma once

template<typename ElemT>
  __global__ void AddReduceEarlyTerm(ElemT* partSum, ElemT* data, size_t dataSize);

#include "../Library/EarlyTerm.cu"