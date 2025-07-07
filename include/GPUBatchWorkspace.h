#pragma once

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <iostream>
#include <vector>

class GPUBatchWorkspace {
public:
    GPUBatchWorkspace(int nRHS, int matrixSize);
    ~GPUBatchWorkspace();

    void ensureSize(int M, int N, int R, int tasknum );

    cuComplex* getDenseMat() const { return d_denseMat_; }
    cuComplex* getQmat() const { return d_Qmat_; }
    cuComplex* getRmat() const { return d_Rmat_; }
    cuComplex* getLocalB() const { return d_localB_; }
    cuComplex* getLocalC() const { return d_localC_; }
    cuComplex* getLocalMat() const { return d_localMat_; }

    cuComplex* getPinnedTempB() const { return h_tempB_; }
    cuComplex* getPinnedTempC() const { return h_tempC_; }

    cuComplex* getGlobalMatB() { return h_globalMatB_; }
    cuComplex* getGlobalMatC() { return h_globalMatC_; }

    cuComplex** getDenseMatArray() const { return d_denseMat_array; }
    cuComplex** getQmatArray() const { return d_Qmat_array; }
    cuComplex** getRmatArray() const { return d_Rmat_array; }
    cuComplex** getLocalBArray() const { return d_localB_array; }
    cuComplex** getLocalCArray() const { return d_localC_array; }   
    cuComplex** getLocalMatArray() const { return d_localMat_array; }

    void setGlobalMatB(cuComplex* matB) { h_globalMatB_ = matB; }

    size_t getAvailableGPUMemory() const;

    void setStream(cudaStream_t stream) { m_stream = stream; }
    cudaStream_t getStream() const { return m_stream; }

    void printMemoryInfo() const;
    void releaseLocalMats();

private:
    cudaStream_t m_stream;
    int m_nRHS;
    int m_matrixSize;
    int m_tasknum = 0; // Number of tasks in the batch

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

    size_t m_denseMatSize = 0;
    size_t m_localMatSize = 0;
    size_t m_localBSize = 0;
    size_t m_localCSize = 0;
    size_t m_QmatSize = 0;
    size_t m_RmatSize = 0;

    void allocateGlobalMats();
    void releaseGlobalMats();
};
