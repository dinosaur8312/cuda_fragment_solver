#include "GEMMSolver.h"
#include <cublas_v2.h>
#include <thrust/device_ptr.h>
#include <thrust/gather.h>
#include <thrust/for_each.h>
#include <thrust/execution_policy.h>

GEMMSolver::GEMMSolver()
    : m_handle(nullptr)
{
    cublasCreate(&m_handle);
}

GEMMSolver::~GEMMSolver()
{
    std::cout << "Destroying GEMMSolver..." << std::endl;
  //  cudaStreamSynchronize(m_stream); // Ensure all operations on the stream are completed before cleanup
    cublasDestroy(m_handle);
}

void GEMMSolver::setStream(cudaStream_t stream)
{
    m_stream = stream;
    cublasSetStream(m_handle, m_stream);
}

__global__ void verify_pointers_kernel(const cuComplex *A, const cuComplex *B, cuComplex *C)
{
    //print first element value
    printf("GEMMSolver::verify_pointers_kernel: A: %p, B: %p, C: %p\n", (void*)A, (void*)B, (void*)C);
    //print values of A, B, C
    printf("GEMMSolver::verify_pointers_kernel: A: %f + %fi, B: %f + %fi, C: %f + %fi\n", A[0].x, A[0].y, B[0].x, B[0].y, C[0].x, C[0].y);
}

void GEMMSolver::gemm(const cuComplex *A, const cuComplex *B, cuComplex *C,
                      int M, int K, int N)
{
    const cuComplex alpha = make_cuComplex(1.0f, 0.0f);
    const cuComplex beta = make_cuComplex(0.0f, 0.0f);
    //print all array pointers
    //printf("GEMMSolver::gemm: A: %p, B: %p, C: %p\n", (void*)A, (void*)B, (void*)C);
    printf("GEMMSolver::gemm: M: %d, K: %d, N: %d\n", M, K, N);


    cublasCgemm(m_handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                M, N, K,
                &alpha,
                A, M,
                B, K,
                &beta,
                C, M);

}

__global__ void gather_kernel(const int *d_gather_map,
                              const cuComplex *d_in,
                              cuComplex *d_out,
                              const int ldin,
                              const int ldout)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int rep = blockIdx.y;

    if (idx < ldin)
    {
        int gather_idx = d_gather_map[idx];
        d_out[gather_idx + rep * ldout] = d_in[idx + rep * ldin];
    }
}

void GEMMSolver::gather(const int *d_gather_map,
                        const cuComplex *d_in,
                        cuComplex *d_out,
                        const int ldin,
                        const int ldout,
                        const int reps)
{
    dim3 block(256);
    dim3 grid(iDivUp(ldin, block.x), reps);
    gather_kernel<<<grid, block, 0, m_stream>>>(d_gather_map, d_in, d_out, ldin, ldout);
}

__global__ void scatter_kernel(const int *d_scatter_map,
                               const cuComplex *d_in,
                               cuComplex *d_out,
                               const int ldin,
                               const int ldout)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int rep = blockIdx.y;

    if (idx < ldin)
    {
        int scatter_idx = d_scatter_map[idx];
        d_out[scatter_idx + rep * ldout] += d_in[idx + rep * ldin];
    }
}

void GEMMSolver::scatter(const int *d_scatter_map, 
                         const cuComplex *d_in,
                         cuComplex *d_out,
                         const int ldin,
                         const int ldout,
                         const int reps)
{
    int blockSize = 256;
    dim3 block(blockSize);
    dim3 grid(iDivUp(ldin, blockSize), reps);

    scatter_kernel<<<grid, block, 0, m_stream>>>(
        d_scatter_map, d_in, d_out, ldin, ldout);
}
