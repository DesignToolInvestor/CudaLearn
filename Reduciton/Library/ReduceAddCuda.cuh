/*
  R e d u c e A d d C u d a . c u h
*/

#pragma once

template<typename ElemT>
  void ReduceAdd(
    ElemT& result, const ElemT* inArray, size_t numElem, unsigned threadPerBlock);

#include "../Library/ReduceAddCuda.cu"