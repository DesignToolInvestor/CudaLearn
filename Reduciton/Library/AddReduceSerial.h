/*
  R e d u c e A d d S e r i a l . h
*/

#pragma once

#include <span>

template<typename ElemT, typename AccumT = ElemT>
AccumT ReduceAddCpu(std::span<const ElemT> data)
{
  AccumT partSum = 0;
  for (const ElemT& elem : data)
    partSum += (AccumT)elem;

  return partSum;
}

//template<typename ElemT, typename AccumT = ElemT>
//AccumT ReduceAddGpu(data)
//{
//  AccumT partSum = 0;
//  for (const ElemT& elem : data)
//    partSum += (AccumT)elem;
//
//  return partSum;
//}