/*
  R e d u c e A d d S e r i a l . h
*/

#pragma once

#include <span>

template<typename ElemT, typename AccumT = ElemT>
AccumT ReduceAddCpu(ElemT* data, unsigned numElem)
{
  AccumT partSum = 0;
  for (unsigned i{ 0 }; i < numElem; i++)
    partSum += (AccumT)data[i];

  return partSum;
}