/*
  R e d u c e A d d . h
*/

#pragma once

#include <span>

template<typename ElemT, typename AccumT = ElemT>
AccumT ReduceAdd(ElemT* data, unsigned numElem)
{
  AccumT partSum = 0;
  for (unsigned i{ 0 }; i < numElem; i++)
    partSum += (AccumT)data[i];

  return partSum;
}