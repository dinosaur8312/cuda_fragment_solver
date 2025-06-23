#include "WorkspaceManager.h"
#include <cuda_runtime_api.h>

WorkspaceManager::WorkspaceManager(int nRHS)
    : m_nRHS(nRHS), maxM_(0), maxN_(0), maxR_(0),
      d_localB_(nullptr), d_localC_(nullptr), d_localMat_(nullptr) {}

WorkspaceManager::~WorkspaceManager() {
    release();
}

void WorkspaceManager::release() {
    if (d_localB_) cudaFree(d_localB_);
    if (d_localC_) cudaFree(d_localC_);
    if (d_localMat_) cudaFree(d_localMat_);
    d_localB_ = d_localC_ = d_localMat_ = nullptr;
}

void WorkspaceManager::allocate() {
    size_t sizeB = maxN_ * m_nRHS * sizeof(cuComplex);
    size_t sizeC = maxM_ * m_nRHS * sizeof(cuComplex);
    size_t sizeMat = maxR_ * m_nRHS * sizeof(cuComplex);

    cudaMalloc(&d_localB_, sizeB);
    cudaMalloc(&d_localC_, sizeC);
    cudaMalloc(&d_localMat_, sizeMat);
}

void WorkspaceManager::ensureSize(int M, int N, int R) {
    bool needResize = false;
    if (M > maxM_) { maxM_ = M; needResize = true; }
    if (N > maxN_) { maxN_ = N; needResize = true; }
    if (R > maxR_) { maxR_ = R; needResize = true; }

    if (needResize) {
        release();
        allocate();
    }
}
