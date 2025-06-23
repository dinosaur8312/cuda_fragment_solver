#pragma once

#include <vector>
#include <cuda_runtime.h>
#include <cuComplex.h>

class GEMMSolver;
class WorkspaceManager;

class GPUTask {
public:
    GPUTask(int id, int M, int N, int R, int matrixSize, int nRHS);
    ~GPUTask();

    void execute(GEMMSolver* solver, WorkspaceManager& ws);

    int M() const { return m_M; }
    int N() const { return m_N; }
    int R() const { return m_R; }
    int id() const { return m_id; }

    void setStream(cudaStream_t stream) { m_stream = stream; }
    cudaStream_t getStream() const { return m_stream; }

private:
    void generateRandomMaps(int matrixSize);
    void generateRandomMatrices();

    void uploadToDevice();

    int m_M, m_N, m_R, m_id;
    int m_matrixSize;
    int m_nRHS;

    std::vector<int> m_srcMap, m_sinkMap;
    std::vector<cuComplex> m_denseMat, m_Qmat, m_Rmat;

    int* d_srcMap = nullptr;
    int* d_sinkMap = nullptr;
    cuComplex* d_denseMat = nullptr;
    cuComplex* d_Qmat = nullptr;
    cuComplex* d_Rmat = nullptr;

    cudaStream_t m_stream;
};
