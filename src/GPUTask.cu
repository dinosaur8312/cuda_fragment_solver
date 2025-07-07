#include "GPUTask.h"
#include "GEMMSolver.h"
#include "GPUWorkspace.h"
//#include <thrust/device_ptr.h>
//#include <thrust/gather.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <helper_cuda.h>
#include <cstdlib>
#include <cassert>

GPUTask::GPUTask(int id, int M, int N, int R, int matrixSize, int nRHS)
    : m_id(id), m_M(M), m_N(N), m_R(R), m_matrixSize(matrixSize), m_nRHS(nRHS), m_stream(nullptr)
{
    //printf("Creating GPUTask with ID: %d, M: %d, N: %d, R: %d, matrixSize: %d, nRHS: %d\n",
    //       m_id, m_M, m_N, m_R, m_matrixSize, m_nRHS);

}

GPUTask::~GPUTask()
{
    std::cout << "Destroying GPUTask with ID: " << m_id << std::endl;

    d_srcMap = nullptr;
    d_sinkMap = nullptr;
    d_denseMat = nullptr;
    d_Qmat = nullptr;
    d_Rmat = nullptr;
}

void GPUTask::generateRandomMaps(int matrixSize)
{
    m_srcMap.resize(m_N);
    m_sinkMap.resize(m_M);

    for (int i = 0; i < m_N; ++i)
        m_srcMap[i] = rand() % matrixSize;

    for (int i = 0; i < m_M; ++i)
        m_sinkMap[i] = rand() % matrixSize;
}

void GPUTask::generateRandomMatrices()
{
    //set all value to one for debugging
    if (m_R == 0)
    {
        m_denseMat.resize(m_M * m_N);
        for (auto &val : m_denseMat)
            val = make_cuComplex(1.0f, 0.0f);
            //val = make_cuComplex(static_cast<float>(rand()) / RAND_MAX, static_cast<float>(rand()) / RAND_MAX);
    }
    else
    {
        m_Qmat.resize(m_M * m_R);
        m_Rmat.resize(m_R * m_N);
        for (auto &val : m_Qmat)
            val = make_cuComplex(1.0f, 0.0f);
            //val = make_cuComplex(static_cast<float>(rand()) / RAND_MAX, static_cast<float>(rand()) / RAND_MAX);
            //val = make_cuComplex(static_cast<float>(rand()) / RAND_MAX, static_cast<float>(rand()) / RAND_MAX);
        for (auto &val : m_Rmat)
            val = make_cuComplex(1.0f, 0.0f);
            //val = make_cuComplex(static_cast<float>(rand()) / RAND_MAX, static_cast<float>(rand()) / RAND_MAX);
            //val = make_cuComplex(static_cast<float>(rand()) / RAND_MAX, static_cast<float>(rand()) / RAND_MAX);
    }
}

void GPUTask::uploadToDevice()
{
    //cudaMalloc(&d_srcMap, sizeof(int) * m_N);
    //cudaMemcpyAsync(d_srcMap, m_srcMap.data(), sizeof(int) * m_N, cudaMemcpyHostToDevice, m_stream);

  //  cudaMalloc(&d_sinkMap, sizeof(int) * m_M);
//    cudaMemcpyAsync(d_sinkMap, m_sinkMap.data(), sizeof(int) * m_M, cudaMemcpyHostToDevice, m_stream);

    //printf("m_stream = %p\n", m_stream);
    //checkCudaErrors( cudaStreamSynchronize(m_stream));
    //printf("m_R = %d, m_M = %d, m_N = %d\n", m_R, m_M, m_N);
    if (m_R == 0)
    {
       // printf("d_denseMat = %p, m_denseMat.size() = %zu\n", d_denseMat, m_denseMat.size());
        size_t sizeDenceMat = m_M * m_N * sizeof(cuComplex);
        checkCudaErrors(cudaMemcpyAsync(d_denseMat, m_denseMat.data(), sizeDenceMat, cudaMemcpyDefault, 0));
    }
    else
    {
       // printf("d_Qmat = %p, m_Qmat.size() = %zu\n", d_Qmat, m_Qmat.size());
       // printf("d_Rmat = %p, m_Rmat.size() = %zu\n", d_Rmat, m_Rmat.size());
        size_t sizeQmat = m_M * m_R * sizeof(cuComplex);
        size_t sizeRmat = m_R * m_N * sizeof(cuComplex);
        //cuComplex *h_test = (cuComplex *)malloc(sizeQmat);
       // cudaDeviceSynchronize();
        //cuComplex *d_test = nullptr;
        //checkCudaErrors(cudaMalloc(&d_test, sizeQmat));
        //checkCudaErrors( cudaMemcpyAsync(d_Qmat, h_test, sizeQmat, cudaMemcpyHostToDevice, m_stream));
        //checkCudaErrors( cudaMemcpy(d_Qmat, h_test, 4, cudaMemcpyDefault));
        //d_Qmat_kernel<<<1, 1 >>>(d_Qmat);
        //checkCudaErrors( cudaMemcpy(h_test, d_test, 4, cudaMemcpyDefault));
        //cudaDeviceSynchronize();
        //exit(0);
        checkCudaErrors( cudaMemcpyAsync(d_Qmat, m_Qmat.data(), sizeQmat, cudaMemcpyHostToDevice, m_stream));
        checkCudaErrors( cudaMemcpyAsync(d_Rmat, m_Rmat.data(), sizeRmat,cudaMemcpyHostToDevice, m_stream));
    }
}
/*
void GPUTask::gather()
{
    assert(m_solver != nullptr && m_workspace != nullptr);
    GEMMSolver *solver = m_solver;
    GPUWorkspace *ws = m_workspace;

    cuComplex *d_localB = getLocalB();
    cuComplex *h_tempB = getPinnedTempB();

    //step1: copy data from fake global matrix to h_tempB using m_srcMap
    for (int i = 0; i < m_nRHS; ++i)
    {
    }

    //gather data from h_tempB to d_localB using d_srcMap
    thrust::device_ptr<int> d_map_ptr(d_srcMap);
    thrust::device_ptr<cuComplex> d_localB_ptr(d_localB);
    for (int i = 0; i < m_nRHS; ++i)
    {
        cudaMemcpyAsync(h_tempB, h_tempB + i * m_matrixSize, sizeof(cuComplex) * m_matrixSize, cudaMemcpyDeviceToHost, m_stream);
        thrust::gather(thrust::host, d_map_ptr, d_map_ptr + m_N, h_tempB, h_tempB + i * m_N);       
        cudaMemcpyAsync(d_localB + i * m_N, h_tempB + i * m_N, sizeof(cuComplex) * m_N, cudaMemcpyHostToDevice, m_stream);  
    }
}
*/
void GPUTask::execute()
{
    //assert(m_solver != nullptr && m_workspace != nullptr);
    assert( m_workspace != nullptr);
    //GEMMSolver *solver = m_solver;
    GPUWorkspace *ws = m_workspace;
    GEMMSolver *solver = m_solver;

  //  ws->printMemoryInfo();

    generateRandomMaps(m_matrixSize);
    //printf("Random maps generated for task ID: %d\n", m_id);
    generateRandomMatrices();
    cudaDeviceSynchronize();
    //printf("Random matrices generated for task ID: %d\n", m_id);
    uploadToDevice();
    gather();
    solve();
    scatter();
    //printf("Task %d finished.\n", m_id);
    /*
    solver->gather(d_srcMap, d_denseMat, d_localB, m_N, m_M, m_nRHS, m_stream);

    if (m_R == 0)
    {
        solver->gemm(d_denseMat, d_localB, d_localC, m_M, m_N, m_nRHS, m_stream);
    }
    else
    {
        solver->gemm(d_Rmat, d_localB, d_localMat, m_R, m_N, m_nRHS, m_stream);
        solver->gemm(d_Qmat, d_localMat, d_localC, m_M, m_R, m_nRHS, m_stream);
    }

    solver->scatterAdd(d_sinkMap, d_localC, m_M, m_nRHS, m_matrixSize, m_stream);
    */
}

void GPUTask::gather()
{
    //gather data from h_globalB to h_tempB using m_srcMap
    for(int j=0; j<m_nRHS; j++)
    {
        for(int i=0; i<m_N; i++)
        {
            h_tempB[j * m_N + i] = h_globalMatB_[m_srcMap[i] + j * m_matrixSize];
        }
    }

    size_t gatherSize = sizeof(cuComplex) * m_nRHS * m_N;

    cudaMemcpyAsync(d_localB, h_tempB, gatherSize, cudaMemcpyHostToDevice, m_stream);
}

void GPUTask::scatter()
{
    cudaMemcpyAsync(h_tempC, d_localC, sizeof(cuComplex) * m_M * m_nRHS, cudaMemcpyDeviceToHost, m_stream);

    //scatter data from d_localC to h_globalC using m_sinkMap
    for(int j=0; j<m_nRHS; j++)
    {
        for(int i=0; i<m_M; i++)
        {
            h_globalMatC_[m_sinkMap[i] + j * m_matrixSize] += h_tempC[j * m_M + i];
        }
    }

}



void GPUTask::setWorkspace(GPUWorkspace *workspace)
{
    m_workspace = workspace;
    // Ensure workspace is allocated for the required sizes
    m_workspace->ensureSize(m_M, m_N, m_R);
    //cudaDeviceSynchronize();
    h_tempB = m_workspace->getPinnedTempB();
    h_tempC = m_workspace->getPinnedTempC();
    d_localB = m_workspace->getLocalB();
    d_localC = m_workspace->getLocalC();
    d_localMat = m_workspace->getLocalMat();
    h_globalMatB_ = m_workspace->getGlobalMatB();
    h_globalMatC_ = m_workspace->getGlobalMatC();
    d_Qmat = m_workspace->getQmat();
    d_Rmat = m_workspace->getRmat();
    /*
    printf("GPUTask::setWorkspace: h_tempB: %p\n", h_tempB);
    printf("GPUTask::setWorkspace: h_globalMatB_: %p\n", h_globalMatB_);
    printf("GPUTask::setWorkspace: d_localB: %p\n", d_localB);
    printf("GPUTask::setWorkspace: d_localC: %p\n", d_localC);
    printf("GPUTask::setWorkspace: d_localMat: %p\n", d_localMat);
    printf("GPUTask::setWorkspace: h_globalMatC_: %p\n", h_globalMatC_);
    printf("GPUTask::setWorkspace: d_Qmat: %p\n", d_Qmat);
    printf("GPUTask::setWorkspace: d_Rmat: %p\n", d_Rmat);
    */
}

void GPUTask::solve()
{
    assert(m_solver != nullptr && m_workspace != nullptr);
    GPUWorkspace *ws = m_workspace;
    GEMMSolver *solver = m_solver;

    //printf("GPUTask::solve: d_denseMat: %p, d_Qmat: %p, d_Rmat: %p\n", d_denseMat, d_Qmat, d_Rmat);
    //printf("GPUTask::solve: m_M: %d, m_N: %d, m_R: %d\n", m_M, m_N, m_R);

    if (m_R == 0)
    {
        solver->gemm(d_denseMat, d_localB, d_localC, m_M, m_N, m_nRHS);
    }
    else
    {
        solver->gemm(d_Rmat, d_localB, d_localMat, m_R, m_N, m_nRHS);
        solver->gemm(d_Qmat, d_localMat, d_localC, m_M, m_R, m_nRHS);
    }

    //cudaDeviceSynchronize();
    //exit(0);
}