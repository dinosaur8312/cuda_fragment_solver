#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <cuComplex.h>
#include <cassert>
#include <algorithm>

class WorkspaceManager {
public:
    WorkspaceManager(int initialMaxM, int initialMaxN, int initialMaxR, int fixedRHS)
        : maxM_(initialMaxM), maxN_(initialMaxN), maxR_(initialMaxR), nRHS_(fixedRHS) {
        allocate();
    }

    ~WorkspaceManager() {
        release();
    }

    void ensureSize(int M, int N, int R) {
        bool needRealloc = false;
        if (M > maxM_ || N > maxN_ || R > maxR_) {
            maxM_ = std::max(maxM_, M);
            maxN_ = std::max(maxN_, N);
            maxR_ = std::max(maxR_, R);
            release();
            allocate();
        }
    }

    cuComplex* getLocalB() const { return d_localB_; }
    cuComplex* getLocalC() const { return d_localC_; }
    cuComplex* getLocalMat() const { return d_localMat_; }

private:
    void allocate() {
        size_t sizeLocalB   = maxN_ * nRHS_ * sizeof(cuComplex);
        size_t sizeLocalC   = maxM_ * nRHS_ * sizeof(cuComplex);
        size_t sizeLocalMat = maxR_ * nRHS_ * sizeof(cuComplex);

        cudaMalloc(&d_localB_, sizeLocalB);
        cudaMalloc(&d_localC_, sizeLocalC);
        cudaMalloc(&d_localMat_, sizeLocalMat);
    }

    void release() {
        if (d_localB_) cudaFree(d_localB_); d_localB_ = nullptr;
        if (d_localC_) cudaFree(d_localC_); d_localC_ = nullptr;
        if (d_localMat_) cudaFree(d_localMat_); d_localMat_ = nullptr;
    }

    int maxM_, maxN_, maxR_, nRHS_;
    cuComplex *d_localB_ = nullptr;
    cuComplex *d_localC_ = nullptr;
    cuComplex *d_localMat_ = nullptr;
};
