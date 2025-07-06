#pragma once

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <vector>
#include <cublas_v2.h>
#include <helper_math.h>

class GEMMSolver {
public:
    GEMMSolver(cudaStream_t stream = 0);
    ~GEMMSolver();

    void gemm(const cuComplex* A, const cuComplex* B, cuComplex* C,
              int M, int K, int N);

    void gather(const int* d_gather_map, 
                const cuComplex* d_in,
                cuComplex* d_out,
                const int ldin,
                const int ldout,
                const int reps);

    void scatter(const int* d_scatter_map, 
                 const cuComplex* d_in,
                 cuComplex* d_out,
                 const int ldin,
                  const int ldout,
                 const int reps);



private:
    cublasHandle_t m_handle;
    cudaStream_t m_stream;
};
