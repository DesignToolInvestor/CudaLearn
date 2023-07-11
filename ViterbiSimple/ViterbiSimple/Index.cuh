/*
  I n d e x . c u h
*/

#pragma once

template<typename IndexT, unsigned innerSize>
IndexT Index2(IndexT outer, IndexT inner)
{
  return outer * innerSize + inner;
}

template<typename IndexT, unsigned size0, unsigned size1>
IndexT Index3(IndexT index2, IndexT index1, IndexT index0)
{
  return ((index2 * size1) + index1) * size0 + index0;
}