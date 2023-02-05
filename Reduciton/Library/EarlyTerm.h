/*
  E a r l y T e r m . h
*/

#pragma once

template<typename ElemT>
  void ReduceAddGpu(
    ElemT& result, const ElemT* data, size_t numElem, unsigned threadPerBlock);