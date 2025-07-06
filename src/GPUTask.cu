#include "GPUTask.h"
#include "GEMMSolver.h"
#include "GPUWorkspace.h"
#include <thrust/device_ptr.h>
#include <thrust/gather.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cassert>

GPUTask::GPUTask(int id, int M, int N, int R, int matrixSize, int nRHS)
    : m_id(id), m_M(M), m_N(N), m_R(R), m_matrixSize(matrixSize), m_nRHS(nRHS), m_stream(nullptr)
{
    //printf("Creating GPUTask with ID: %d, M: %d, N: %d, R: %d, matrixSize: %d, nRHS: %d\n",
    //       m_id, m_M, m_N, m_R, m_matrixSize, m_nRHS);
    generateRandomMaps(matrixSize);
    //printf("Random maps generated for task ID: %d\n", m_id);
    generateRandomMatrices();
    //printf("Random matrices generated for task ID: %d\n", m_id);
    uploadToDevice();
}

GPUTask::~GPUTask()
{
    std::cout << "Destroying GPUTask with ID: " << m_id << std::endl;
    if (d_srcMap)
        cudaFree(d_srcMap);
    if (d_sinkMap)
        cudaFree(d_sinkMap);
    if (d_denseMat)
        cudaFree(d_denseMat);
    if (d_Qmat)
        cudaFree(d_Qmat);
    if (d_Rmat)
        cudaFree(d_Rmat);

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
    if (m_R == 0)
    {
        m_denseMat.resize(m_M * m_N);
        for (auto &val : m_denseMat)
            val = make_cuComplex(static_cast<float>(rand()) / RAND_MAX, static_cast<float>(rand()) / RAND_MAX);
    }
    else
    {
        m_Qmat.resize(m_M * m_R);
        m_Rmat.resize(m_R * m_N);
        for (auto &val : m_Qmat)
            val = make_cuComplex(static_cast<float>(rand()) / RAND_MAX, static_cast<float>(rand()) / RAND_MAX);
        for (auto &val : m_Rmat)
            val = make_cuComplex(static_cast<float>(rand()) / RAND_MAX, static_cast<float>(rand()) / RAND_MAX);
    }
}

void GPUTask::uploadToDevice()
{
    cudaMalloc(&d_srcMap, sizeof(int) * m_N);
    cudaMemcpyAsync(d_srcMap, m_srcMap.data(), sizeof(int) * m_N, cudaMemcpyHostToDevice, m_stream);

    cudaMalloc(&d_sinkMap, sizeof(int) * m_M);
    cudaMemcpyAsync(d_sinkMap, m_sinkMap.data(), sizeof(int) * m_M, cudaMemcpyHostToDevice, m_stream);

    if (m_R == 0)
    {
        cudaMalloc(&d_denseMat, sizeof(cuComplex) * m_M * m_N);
        cudaMemcpyAsync(d_denseMat, m_denseMat.data(), sizeof(cuComplex) * m_M * m_N, cudaMemcpyHostToDevice, m_stream);
    }
    else
    {
        cudaMalloc(&d_Qmat, sizeof(cuComplex) * m_M * m_R);
        cudaMemcpyAsync(d_Qmat, m_Qmat.data(), sizeof(cuComplex) * m_M * m_R, cudaMemcpyHostToDevice, m_stream);

        cudaMalloc(&d_Rmat, sizeof(cuComplex) * m_R * m_N);
        cudaMemcpyAsync(d_Rmat, m_Rmat.data(), sizeof(cuComplex) * m_R * m_N, cudaMemcpyHostToDevice, m_stream);
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

    cuComplex *d_localB = ws->getLocalB(m_N);
    cuComplex *d_localC = ws->getLocalC(m_M);
    cuComplex *d_localMat = ws->getLocalMat(m_R);

    ws->printMemoryInfo();

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


void GPUTask::setWorkspace(GPUWorkspace *workspace)
{
    m_workspace = workspace;
    // Ensure workspace is allocated for the required sizes
    m_workspace->ensureSize(m_M, m_N, m_R);
}