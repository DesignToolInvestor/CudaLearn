/*
  R e d u c e A d d W r a p p e r . h
*/

#pragma once

template<typename ElemT>
  void ReduceAddWrap(
    ElemT& result, const ElemT* inArray, size_t numElems, unsigned threadPerBlock);