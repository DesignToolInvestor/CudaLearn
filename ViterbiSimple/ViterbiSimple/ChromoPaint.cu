/*
  C h r o m o P a i n t . c u
*/

#include "ChromoPaint.cuh"
#include "../../Lib/ErrorCheck.cuh"

template<typename ElemT>
using ChromPaintKernT = void (*)(
  unsigned numPeople, unsigned numHap, unsigned numSnp,
  const uint8_t read[], const uint8_t hap[], HapIndexT path[],
  const ProbT StateStayProb[], const ProbT hapWeight,
  unsigned threadPerBlock);

template<
  typename ProbT, typename PeopleIndexT, typename HapIndexT, typename SnpIndexT>
void ChromoPaintUnPhase(
  unsigned numPeople, unsigned numSnp,
  const uint8_t read[], PeopleIndexT path[],
  const ProbT stateStayProb[],
  unsigned threadPerBlock, unsigned enoughThreads)
{
  // Parameter processing
  const size_t readElems = numPeople * numSnp;
  const size_t readBytes = readElems * sizeof(uint8_t);

  const size_t pathElems = numPeople * numSnp;
  const size_t pathElems = PathElems * sizeof(HapIndexT);

  // Choose which GPU to run on, change this on a multi-GPU system.
  CheckErr(cudaSetDevice(0), "No cuda devices.");

  // Get device properties
  cudaDeviceProp devProp;
  CheckErr(cudaGetDeviceProperties(&devProp, 0), "Can't get device properties");

  // Allocate GPU buffers for read, hap, and path
  uint8_t* readDev = NULL;
  HapIndexT* hapDev = NULL;

  CheckErr(cudaMalloc((void**)&readDev, readBytes), "Read array allocation failed");
  CheckErr(cudaMalloc((void**)&hapDev, hapBytes), "Haplotype array allocation failed");
  CheckErr(cudaMalloc((void**)&pathDev, pathBytes), "Path array allocation failed");

  // Copy input vectors from host memory to GPU buffers.
  CheckErr(cudaMemcpy(readDev, read, readBytes, cudaMemcpyHostToDevice), "Copying read failed");
  CheckErr(cudaMemcpy(hapDev, hap, hapBytes, cudaMemcpyHostToDevice), "Copying read failed");

  // Launch kernel
  unsigned numThread = numPeople * numHap;
  unsigned numBlock = (unsigned)((numThread + (threadPerBlock - 1)) / threadPerBlock);

  Kern << <numBlock, threadPerBlock >> > (data_d, dataElems);


}