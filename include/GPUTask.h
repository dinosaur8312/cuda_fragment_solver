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

    void setSrcMap(const std::vector<int> &srcMap) { m_srcMap = srcMap; }
    void setSinkMap(const std::vector<int> &sinkMap) { m_sinkMap = sinkMap; }
    void setDenseMat(const std::vector<cuComplex> &denseMat) { m_denseMat = denseMat; }
    void setQmat(const std::vector<cuComplex> &Qmat) { m_Qmat = Qmat; }
    void setRmat(const std::vector<cuComplex> &Rmat) { m_Rmat = Rmat; }

    void setSolver(GEMMSolver *solver) { m_solver = solver; }
    void setWorkspace(GPUWorkspace *workspace);

    void uploadToDevice();
    void solve();
private:
    void generateRandomMaps(int matrixSize);
    void generateRandomMatrices();


    int m_M, m_N, m_R, m_id;
    int m_matrixSize;
    int m_nRHS;

    std::vector<int> m_srcMap, m_sinkMap;
    std::vector<cuComplex> m_denseMat, m_Qmat, m_Rmat;

    int *d_srcMap = nullptr;
    int *d_sinkMap = nullptr;
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
