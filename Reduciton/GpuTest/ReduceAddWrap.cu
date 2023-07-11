/*
  R e d u c e A d d W r a p . c u
*/

#include "../Library/ReduceAddCuda.cuh"
#include "../Library/EarlyTerm.cuh"

// Create wrapper
template<typename ElemT>
void ReduceAddWrap(
  ElemT& result, const ElemT* inArray, size_t numElems, unsigned threadPerBlock)
{
  ReduceAddCuda<ElemT>(result, inArray, numElems, threadPerBlock);
}

// ************************************
// Create the instantiations used by this project
template void ReduceAddCuda<int>(
  int& result, const int* inArray, size_t inElemsArg, unsigned threadPerBlock);

template void ReduceAddWrap<int>(
  int& result, const int* inArray, size_t numElems, unsigned threadPerBlock);