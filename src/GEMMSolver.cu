#include "GEMMSolver.h"
#include <cublas_v2.h>
#include <thrust/device_ptr.h>
#include <thrust/for_each.h>
#include <thrust/execution_policy.h>

GEMMSolver::GEMMSolver() {
    cublasCreate(&m_handle);
}

GEMMSolver::~GEMMSolver() {
    cublasDestroy(m_handle);
}

void GEMMSolver::gemm(const cuComplex* A, const cuComplex* B, cuComplex* C,
                      int M, int K, int N, cudaStream_t stream) {
    const cuComplex alpha = make_cuComplex(1.0f, 0.0f);
    const cuComplex beta = make_cuComplex(0.0f, 0.0f);

    cublasSetStream(m_handle, stream);
    cublasCgemm(m_handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                M, N, K,
                &alpha,
                A, M,
                B, K,
                &beta,
                C, M);
}

void GEMMSolver::gather(const int* d_gather_map, cuComplex* d_out,
                        int num_rows, int num_cols, int matrixSize, cudaStream_t stream) {
    size_t total = static_cast<size_t>(num_rows) * num_cols;

    for (int j = 0; j < num_cols; ++j) {
        thrust::gather(thrust::cuda::par.on(stream),
                       d_gather_map,
                       d_gather_map + num_rows,
                       thrust::device_pointer_cast(d_out + matrixSize * j),
                       thrust::device_pointer_cast(d_out + num_rows * j));
    }
}

__global__ void scatter_add_kernel(const int* map, cuComplex* global, const cuComplex* local,
                                   int num_rows, int num_cols, int matrixSize) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y;

    if (row < num_rows && col < num_cols) {
        int global_idx = map[row] + matrixSize * col;
        int local_idx = row + num_rows * col;

        atomicAdd(&global[global_idx].x, local[local_idx].x);
        atomicAdd(&global[global_idx].y, local[local_idx].y);
    }
}

void GEMMSolver::scatterAdd(const int* d_scatter_map, const cuComplex* d_in,
                            int num_rows, int num_cols, int matrixSize, cudaStream_t stream) {
    int blockSize = 256;
    dim3 block(blockSize);
    dim3 grid((num_rows + blockSize - 1) / blockSize, num_cols);

    scatter_add_kernel<<<grid, block, 0, stream>>>(
        d_scatter_map, d_in /* global output */, d_in, num_rows, num_cols, matrixSize);
}
