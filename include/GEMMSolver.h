#pragma once

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <vector>

class GEMMSolver {
public:
    GEMMSolver();
    ~GEMMSolver();

    void gemm(const cuComplex* A, const cuComplex* B, cuComplex* C,
              int M, int K, int N, cudaStream_t stream);

    void gather(const int* d_gather_map, cuComplex* d_out,
                int num_rows, int num_cols, int matrixSize, cudaStream_t stream);

    void scatterAdd(const int* d_scatter_map, const cuComplex* d_in,
                    int num_rows, int num_cols, int matrixSize, cudaStream_t stream);

private:
    cublasHandle_t m_handle;
};
