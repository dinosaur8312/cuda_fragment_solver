#pragma once

#include "GPUTask.h"
#include <tuple>

class GPUBatchWorkspace;
class GEMMBatchSolver;

class GPUBatchTask
{
public:
    GPUBatchTask(int M_pad, int N_pad, int R_pad);
    GPUBatchTask(const std::tuple<int, int, int>& dims)
        : GPUBatchTask(std::get<0>(dims), std::get<1>(dims), std::get<2>(dims)) {}
    ~GPUBatchTask();
    void addTask(GPUTask* task);
    std::vector<GPUTask*>& getTasks() { return m_tasks; }
    std::tuple<int, int, int> getDims() { return std::make_tuple(m_M_pad, m_N_pad, m_R_pad); }

    void setStream(cudaStream_t stream) { m_stream = stream; }
    void setWorkspace(GPUBatchWorkspace* workspace);
    void setSolver(GEMMBatchSolver* solver) { m_solver = solver; }
    void execute();
private:
    int m_M_pad, m_N_pad, m_R_pad;
    int m_matrixSize;
    int m_nRHS;
    std::vector<GPUTask*> m_tasks;
    cudaStream_t m_stream;

    std::vector<std::vector<int>> m_srcMaps;
    std::vector<std::vector<int>> m_sinkMaps;
    std::vector<cuComplex> m_denseMat;
    std::vector<cuComplex> m_Qmat;
    std::vector<cuComplex> m_Rmat;

    GEMMBatchSolver* m_solver;
    GPUBatchWorkspace* m_workspace;


    cuComplex** d_denseMat_array = nullptr; // Array of device pointers for dense matrices
    cuComplex** d_Qmat_array = nullptr; // Array of device pointers for Q matrices
    cuComplex** d_Rmat_array = nullptr; // Array of device pointers for R matrices
    cuComplex** d_localB_array = nullptr; // Array of device pointers for local B matrices
    cuComplex** d_localC_array = nullptr; // Array of device pointers for local C matrices
    cuComplex** d_localMat_array = nullptr; // Array of device pointers for local

    cuComplex* d_denseMat_ = nullptr;
    cuComplex* d_Qmat_ = nullptr;
    cuComplex* d_Rmat_ = nullptr;
    cuComplex* d_localB_ = nullptr;
    cuComplex* d_localC_ = nullptr;
    cuComplex* d_localMat_ = nullptr;

    // pinned memory
    cuComplex* h_tempB_ = nullptr;
    cuComplex* h_tempC_ = nullptr;

    // non-pinned host global matrix
    cuComplex* h_globalMatB_ = nullptr;
    cuComplex* h_globalMatC_ = nullptr;


    void uploadBatchToDevice();

    void generateRandomMaps(int matrixSize);
    void generateRandomMatrices();
    void uploadToDevice();
    void gather();
    void solve();
    void scatter();
};