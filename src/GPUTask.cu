#include "GPUTask.h"
#include "GEMMSolver.h"
#include "WorkspaceManager.h"
#include <thrust/device_ptr.h>
#include <thrust/gather.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cassert>

GPUTask::GPUTask(int id, int M, int N, int R, int matrixSize, int nRHS)
    : m_id(id), m_M(M), m_N(N), m_R(R), m_matrixSize(matrixSize), m_nRHS(nRHS), m_stream(nullptr) {
    generateRandomMaps(matrixSize);
    generateRandomMatrices();
    uploadToDevice();
}

GPUTask::~GPUTask() {
    if (d_srcMap) cudaFree(d_srcMap);
    if (d_sinkMap) cudaFree(d_sinkMap);
    if (d_denseMat) cudaFree(d_denseMat);
    if (d_Qmat) cudaFree(d_Qmat);
    if (d_Rmat) cudaFree(d_Rmat);
}

void GPUTask::generateRandomMaps(int matrixSize) {
    m_srcMap.resize(m_N);
    m_sinkMap.resize(m_M);

    for (int i = 0; i < m_N; ++i)
        m_srcMap[i] = rand() % matrixSize;

    for (int i = 0; i < m_M; ++i)
        m_sinkMap[i] = rand() % matrixSize;
}

void GPUTask::generateRandomMatrices() {
    if (m_R == 0) {
        m_denseMat.resize(m_M * m_N);
        for (auto& val : m_denseMat)
            val = make_cuComplex(static_cast<float>(rand()) / RAND_MAX, static_cast<float>(rand()) / RAND_MAX);
    } else {
        m_Qmat.resize(m_M * m_R);
        m_Rmat.resize(m_R * m_N);
        for (auto& val : m_Qmat)
            val = make_cuComplex(static_cast<float>(rand()) / RAND_MAX, static_cast<float>(rand()) / RAND_MAX);
        for (auto& val : m_Rmat)
            val = make_cuComplex(static_cast<float>(rand()) / RAND_MAX, static_cast<float>(rand()) / RAND_MAX);
    }
}

void GPUTask::uploadToDevice() {
    cudaMalloc(&d_srcMap, sizeof(int) * m_N);
    cudaMemcpyAsync(d_srcMap, m_srcMap.data(), sizeof(int) * m_N, cudaMemcpyHostToDevice, m_stream);

    cudaMalloc(&d_sinkMap, sizeof(int) * m_M);
    cudaMemcpyAsync(d_sinkMap, m_sinkMap.data(), sizeof(int) * m_M, cudaMemcpyHostToDevice, m_stream);

    if (m_R == 0) {
        cudaMalloc(&d_denseMat, sizeof(cuComplex) * m_M * m_N);
        cudaMemcpyAsync(d_denseMat, m_denseMat.data(), sizeof(cuComplex) * m_M * m_N, cudaMemcpyHostToDevice, m_stream);
    } else {
        cudaMalloc(&d_Qmat, sizeof(cuComplex) * m_M * m_R);
        cudaMemcpyAsync(d_Qmat, m_Qmat.data(), sizeof(cuComplex) * m_M * m_R, cudaMemcpyHostToDevice, m_stream);

        cudaMalloc(&d_Rmat, sizeof(cuComplex) * m_R * m_N);
        cudaMemcpyAsync(d_Rmat, m_Rmat.data(), sizeof(cuComplex) * m_R * m_N, cudaMemcpyHostToDevice, m_stream);
    }
}

void GPUTask::execute(GEMMSolver* solver, WorkspaceManager& ws) {
    cuComplex* d_localB   = ws.getLocalB(m_N, m_nRHS);
    cuComplex* d_localC   = ws.getLocalC(m_M, m_nRHS);
    cuComplex* d_localMat = (m_R > 0) ? ws.getLocalMat(m_R, m_nRHS) : nullptr;

    solver->gather(d_srcMap, d_localB, m_N, m_nRHS, m_matrixSize, m_stream);

    if (m_R == 0) {
        solver->gemm(d_denseMat, d_localB, d_localC, m_M, m_N, m_nRHS, m_stream);
    } else {
        solver->gemm(d_Rmat, d_localB, d_localMat, m_R, m_N, m_nRHS, m_stream);
        solver->gemm(d_Qmat, d_localMat, d_localC, m_M, m_R, m_nRHS, m_stream);
    }

    solver->scatterAdd(d_sinkMap, d_localC, m_M, m_nRHS, m_matrixSize, m_stream);
}
