#pragma once

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <vector>
#include <cublas_v2.h>
#include <helper_math.h>

class GEMMBatchSolver {
public:
    GEMMBatchSolver();
    ~GEMMBatchSolver();

    void gemm(cuComplex* const* A, cuComplex* const* B, cuComplex **C,
                      int M, int K, int N, int batchCount);

    void setStream(cudaStream_t stream);



private:
    cublasHandle_t m_handle;
    cudaStream_t m_stream;
};
