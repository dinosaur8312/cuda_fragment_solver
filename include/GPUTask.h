#pragma once

#include <vector>
#include <cuda_runtime.h>
#include <cuComplex.h>
class GPUWorkspace;
class GEMMSolver;

class GPUTask
{
public:
    GPUTask(int id, int M, int N, int R, int matrixSize, int nRHS);
    ~GPUTask();

    void execute();

    int maxtrixSize() const { return m_matrixSize; }
    int nRHS() const { return m_nRHS; }

    int M() const { return m_M; }
    int N() const { return m_N; }
    int R() const { return m_R; }
    int id() const { return m_id; }

    void setStream(cudaStream_t stream) { m_stream = stream; }
    cudaStream_t getStream() const { return m_stream; }

    const std::vector<int> &getSrcMap() const { return m_srcMap; }
    const std::vector<int> &getSinkMap() const { return m_sinkMap; }
    const std::vector<cuComplex> &getDenseMat() const { return m_denseMat; }
    const std::vector<cuComplex> &getQmat() const { return m_Qmat; }
    const std::vector<cuComplex> &getRmat() const { return m_Rmat; }


    void setSolver(GEMMSolver *solver) { m_solver = solver; }
    void setWorkspace(GPUWorkspace *workspace);

    void uploadToDevice();
    void solve();
    void generateRandomMaps(int matrixSize);
    void generateRandomMatrices();
    void generateRandomPaddedMatrices(int M_pad, int N_pad, int R_pad);
private:


    int m_M, m_N, m_R, m_id;
    int m_matrixSize;
    int m_nRHS;

    std::vector<int> m_srcMap, m_sinkMap;
    std::vector<cuComplex> m_denseMat, m_Qmat, m_Rmat;

    cuComplex *d_denseMat = nullptr;
    cuComplex *d_Qmat = nullptr;
    cuComplex *d_Rmat = nullptr;

    cudaStream_t m_stream;

    GEMMSolver *m_solver = nullptr;
    GPUWorkspace *m_workspace = nullptr;


    cuComplex *h_globalMatB_ = nullptr; // Host global matrix B shared across all threads
    cuComplex *h_globalMatC_ = nullptr; // Host global matrix C shared across all threads
    // Temporary matrices for gather/scatter operations
    cuComplex *d_localB = nullptr;
    cuComplex *d_localC = nullptr;
    cuComplex *d_localMat = nullptr;
    cuComplex *h_tempB = nullptr;
    cuComplex *h_tempC = nullptr;

   /*

    cuComplex *getLocalB()
    {
        return m_workspace ? m_workspace->getLocalB(m_N) : nullptr;
    }
    cuComplex *getLocalC()
    {
        return m_workspace ? m_workspace->getLocalC(m_M) : nullptr;
    }

    cuComplex *getLocalMat()
    {
        return m_workspace ? m_workspace->getLocalMat(m_R) : nullptr;
    }

    cuComplex *getPinnedTempB()
    {
        return m_workspace ? m_workspace->getPinnedTempB(m_N) : nullptr;
    }
    cuComplex *getPinnedTempC()
    {
        return m_workspace ? m_workspace->getPinnedTempC(m_M) : nullptr;
    }
*/
    void gather();
    void scatter();
};
