#pragma once

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <iostream>
#include <vector>

class GPUWorkspace {
public:
    GPUWorkspace(int nRHS, int matrixSize);
    ~GPUWorkspace();

    void ensureSize(int M, int N, int R);

    cuComplex* getDenseMat() const { return d_denseMat; }
    cuComplex* getQmat() const { return d_Qmat; }
    cuComplex* getRmat() const { return d_Rmat; }
    cuComplex* getLocalB() const { return d_localB_; }
    cuComplex* getLocalC() const { return d_localC_; }
    cuComplex* getLocalMat() const { return d_localMat_; }

    cuComplex* getPinnedTempB() const { return h_tempB_; }
    cuComplex* getPinnedTempC() const { return h_tempC_; }

    cuComplex* getGlobalMatB(){ return h_globalMatB_; }
    cuComplex* getGlobalMatC(){ return h_globalMatC_; }

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

    //int maxM_;
    //int maxN_;
    //int maxR_;

    cuComplex *d_denseMat = nullptr;
    cuComplex *d_Qmat = nullptr;
    cuComplex *d_Rmat = nullptr;


    cuComplex* d_localB_;
    cuComplex* d_localC_;
    cuComplex* d_localMat_;

    //pinned memory
    cuComplex* h_tempB_ = nullptr;
    cuComplex* h_tempC_ = nullptr;

    //non-pinned host global matrix
    cuComplex* h_globalMatB_ = nullptr;
    cuComplex* h_globalMatC_ = nullptr;

    size_t m_denseMatSize;
    size_t m_localMatSize;    
    size_t m_localBSize;    
    size_t m_localCSize;    
    size_t m_QmatSize;    
    size_t m_RmatSize;    


    void allocateGlobalMats();
    void releaseGlobalMats();

};
