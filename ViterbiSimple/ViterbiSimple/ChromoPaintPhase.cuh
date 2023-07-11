/*
  C h r o m o P a i n t P h a s e . c u h

    This function starts with a collection of chromosomes and paints each one in terms of every
    other chromosome in the collection.  The chromosomes are represented as a list of SNPs.

    The function doesn't preform phasing.

    It seems that several of the algorithmic details are a bit fuzzy (i.e., maybe slightly wrong),
    but this version is useful for understanding performance bottlenecks.

  ARGUMENTS:
    numPeople - This is the number of chromosomes in the collection.

    numSnp - This is the number of SNPs in each chromosome.

    read - This is the measured values of the SNPs for each chromosome in the collection.  It is a
      array of size numPeople x numSnp.  Although, in the final version it should be a table of
      single bit elements (reference or alternative), at this time it is a table of uint8_t's (just
      to get something working).

    path - This is an output of the painting of each chromosome in terms of the other chromosomes.
      It is an array of size numPeople x numSnp of elements of type PeopleIndexT.

    stateStayProb - This is an array of the probability of NOT changing states between each SNP.
      It is an array of (size numSnp - 1).  The first element is the probability of on NOT changing
      sates between the first and the second SNP, and so on.

    threadPerBlock - This is the number of threads in a GPU block.  It appears that this is a
      tuning parameter that has more to do with the hardware details than with the size of the
      problem.

    enoughThreads - This is the number of threads that expose sufficient parallelism.  If the
      problem is small the function will run all the painting operations in parallel, for
      sufficiently small problems this may still be fewer threads.  For large problem only some of
      the chromosomes would be painted in parallel in order to reduce memory demands.  This
      parameter provides a way for the algorithm to transition between these two cases.

    In the final version there should be a function (or an app note) that helps the user select
    near optimal parameters for the last two arguments.  The optimal values should be worked out
    for the case where this is the only application running on the GPU, through experimentation
    (on all the different hardware configurations).  But when this is running in parallel with
    some other (possibly cooperating) collection of kernels, the partitioning of resources will
    will have to be worked out by the user.  But notes or helper functions should be provided for
    this case.
*/

#pragma once

// Old fashion C types
#include <stdint.h>

template<
  typename ProbT, typename PeopleIndexT, typename SnpIndexT>
void ChromoPaintAllPhase(
  PeopleIndexT path[],
  unsigned numPeople, unsigned numSnp, const uint8_t read[], const ProbT logNoTransProb[],
  unsigned threadPerBlock, unsigned enoughThreads);

template<
  typename ProbT, typename PeopleIndexT, typename SnpIndexT>
void ChromoPaintOnePhase(
  ProbT path[],
  unsigned numPeople, unsigned numSnp, const uint8_t read[], const ProbT logNoTransProb[],
  unsigned toDoStart, unsigned toDoNum);