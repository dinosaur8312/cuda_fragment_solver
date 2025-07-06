#pragma once

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <iostream>
#include <vector>

class GPUWorkspace {
public:
    GPUWorkspace(int nRHS, int matrixSize, cudaStream_t  stream);
    ~GPUWorkspace();

    void ensureSize(int M, int N, int R);

    cuComplex* getLocalB(const int N = -1);
    cuComplex* getLocalC(const int M = -1);
    cuComplex* getLocalMat(const int R = -1);

    cuComplex* getPinnedTempB(const int N = -1);
    cuComplex* getPinnedTempC(const int M = -1);

    cuComplex* getGlobalMatB();
    cuComplex* getGlobalMatC();

    size_t getAvailableGPUMemory() const;

    void setStream(cudaStream_t stream) { m_stream = stream; }
    cudaStream_t getStream() const { return m_stream; }

    void printMemoryInfo() const;

private:

    cudaStream_t m_stream;
    int m_nRHS;
    int m_matrixSize;

    int maxM_;
    int maxN_;
    int maxR_;

    cuComplex* d_localB_;
    cuComplex* d_localC_;
    cuComplex* d_localMat_;

    //pinned memory
    cuComplex* h_tempB_ = nullptr;
    cuComplex* h_tempC_ = nullptr;

    //non-pinned host global matrix
    cuComplex* h_globalMatB_ = nullptr;
    cuComplex* h_globalMatC_ = nullptr;

    void allocateLocalMats();
    void releaseLocalMats();

    void allocateGlobalMats();
    void releaseGlobalMats();

    // Global matrix management
    void ensureGlobalSize(int M, int N, int R);
};
