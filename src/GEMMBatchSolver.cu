#include "GEMMBatchSolver.h"
#include <cublas_v2.h>
#include <thrust/device_ptr.h>
#include <thrust/gather.h>
#include <thrust/for_each.h>
#include <thrust/execution_policy.h>

GEMMBatchSolver::GEMMBatchSolver()
    : m_handle(nullptr)
{
    cublasCreate(&m_handle);
}

GEMMBatchSolver::~GEMMBatchSolver()
{
    std::cout << "Destroying GEMMBatchSolver..." << std::endl;
  //  cudaStreamSynchronize(m_stream); // Ensure all operations on the stream are completed before cleanup
    cublasDestroy(m_handle);
}

void GEMMBatchSolver::setStream(cudaStream_t stream)
{
    m_stream = stream;
    cublasSetStream(m_handle, m_stream);
}

__global__ void verify_pointers_kernel(cuComplex* const* A, cuComplex* const* B, cuComplex **C, const int batchCount)
{
    //print first element value
    //print values of A, B, C
    for(int i = 0; i < batchCount; i++)
    {
        printf("GEMMBatchSolver::verify_pointers_kernel: A[%d]: %p, B[%d]: %p, C[%d]: %p\n",
            i, (void*)A[i],i, (void*)B[i],i, (void*)C[i]);
        printf("GEMMBatchSolver::verify_pointers_kernel: A[%d][0]: (%f, %f), B[%d][0]: (%f, %f), C[%d][0]: (%f, %f)\n",
           i,cuCrealf(A[i][0]), cuCimagf(A[i][0]),
           i,cuCrealf(B[i][0]), cuCimagf(B[i][0]),
           i,cuCrealf(C[i][0]), cuCimagf(C[i][0]));
    }
}
void GEMMBatchSolver::gemm(cuComplex* const* A, cuComplex* const* B, cuComplex **C,
                      int M, int K, int N, int batchCount)
{
    const cuComplex alpha = make_cuComplex(1.0f, 0.0f);
    const cuComplex beta = make_cuComplex(0.0f, 0.0f);
    //print all array pointers
    //printf("GEMMSolver::gemm: A: %p, B: %p, C: %p\n", (void*)A, (void*)B, (void*)C);
    /*
    printf("GEMMBatchSolver::gemm: M: %d, K: %d, N: %d, batchCount: %d \n", M, K, N, batchCount);
    cudaDeviceSynchronize(); // Ensure previous operations are completed before proceeding
    fflush(stdout); // Flush stdout to ensure logs are printed immediately
    verify_pointers_kernel<<<1, 1>>>(A, B, C, batchCount);
    cudaDeviceSynchronize(); // Ensure previous operations are completed before proceeding
    fflush(stdout); // Flush stdout to ensure logs are printed immediately
*/
    // Perform batched GEMM operation
    cublasCgemmBatched(m_handle,
                       CUBLAS_OP_N, CUBLAS_OP_N,
                       M, N, K,
                       &alpha,
                       A, M,
                       B, K,
                       &beta,
                       C, M,
                       batchCount);

}

