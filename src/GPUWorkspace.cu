#include "GPUWorkspace.h"
#include <cuda_runtime_api.h>

GPUWorkspace::GPUWorkspace(int nRHS, int matrixSize, cudaStream_t stream)
    : m_stream(stream), m_nRHS(nRHS), m_matrixSize(matrixSize), maxM_(0), maxN_(0), maxR_(0),
      d_localB_(nullptr), d_localC_(nullptr), d_localMat_(nullptr),
      h_tempB_(nullptr), h_tempC_(nullptr), h_globalMatB_(nullptr), h_globalMatC_(nullptr)
{
    allocateGlobalMats();
}

GPUWorkspace::~GPUWorkspace()
{
    std::cout << "GPUWorkspace::~GPUWorkspace() start" << std::endl;
    cudaStreamSynchronize(m_stream); // Ensure all operations on the stream are completed before cleanup
    releaseLocalMats();
    releaseGlobalMats();
    // If you add global matrix management, releaseGlobal();

    std::cout << "GPUWorkspace::~GPUWorkspace() end" << std::endl;
}

cuComplex *GPUWorkspace::getLocalB(const int N)
{
    if (N < 0)
        return d_localB_;
    if (N > maxN_)
    {
        if (d_localB_)
            cudaFreeAsync(d_localB_, m_stream);
        if (h_tempB_)
            cudaFreeHost(h_tempB_);
        maxN_ = N;
        size_t sizeB = maxN_ * m_nRHS * sizeof(cuComplex);
        cudaMallocAsync(&d_localB_, sizeB, m_stream);
        cudaMallocHost(&h_tempB_, sizeB);
    }
    return d_localB_;
}

cuComplex *GPUWorkspace::getLocalC(const int M)
{
    if (M < 0)
        return d_localC_;
    if (M > maxM_)
    {
        if (d_localC_)
            cudaFreeAsync(d_localC_, m_stream);
        if (h_tempC_)
            cudaFreeHost(h_tempC_);
        maxM_ = M;
        size_t sizeC = maxM_ * m_nRHS * sizeof(cuComplex);
        cudaMallocAsync(&d_localC_, sizeC, m_stream);
        cudaMallocHost(&h_tempC_, sizeC);
    }
    return d_localC_;
}

cuComplex *GPUWorkspace::getLocalMat(const int R)
{
    if (R < 0)
        return d_localMat_;
    if (R > maxR_)
    {
        if (d_localMat_)
            cudaFreeAsync(d_localMat_, m_stream);

        maxR_ = R;
        size_t sizeMat = maxR_ * m_nRHS * sizeof(cuComplex);
        cudaMallocAsync(&d_localMat_, sizeMat, m_stream);
    }
    return d_localMat_;
}

cuComplex *GPUWorkspace::getPinnedTempB(const int N)
{
    if (N < 0)
        return h_tempB_;
    if (N > maxN_)
    {
        if (h_tempB_)
            cudaFreeHost(h_tempB_);
        maxN_ = N;
        size_t sizeB = maxN_ * m_nRHS * sizeof(cuComplex);
        cudaMallocHost(&h_tempB_, sizeB);
    }
    return h_tempB_;
}

cuComplex *GPUWorkspace::getPinnedTempC(const int M)
{
    if (M < 0)
        return h_tempC_;
    if (M > maxM_)
    {
        if (h_tempC_)
            cudaFreeHost(h_tempC_);
        maxM_ = M;
        size_t sizeC = maxM_ * m_nRHS * sizeof(cuComplex);
        cudaMallocHost(&h_tempC_, sizeC);
    }
    return h_tempC_;
}

void GPUWorkspace::allocateGlobalMats()
{
    if (!h_globalMatB_)
    {
        size_t sizeB = m_matrixSize * m_nRHS * sizeof(cuComplex);
        h_globalMatB_ = (cuComplex *)malloc(sizeB);
    }
    if (!h_globalMatC_)
    {
        size_t sizeC = m_matrixSize * m_nRHS * sizeof(cuComplex);
        h_globalMatC_ = (cuComplex *)malloc(sizeC);
    }
}

void GPUWorkspace::releaseGlobalMats()
{
    std::cout << "GPUWorkspace::releaseGlobalMats() start" << std::endl;
    if (h_globalMatB_)
    {
        free(h_globalMatB_);
        h_globalMatB_ = nullptr;
    }
    if (h_globalMatC_)
    {
        free(h_globalMatC_);
        h_globalMatC_ = nullptr;
    }
    std::cout << "GPUWorkspace::releaseGlobalMats() end" << std::endl;
}

void GPUWorkspace::releaseLocalMats()
{
    if (d_localB_)
    {
        cudaError_t err = cudaFreeAsync(d_localB_, m_stream);
        if (err != cudaSuccess)
        {
            std::cerr << "Error freeing d_localB_: " << cudaGetErrorString(err) << std::endl;
        }

    }
    if (d_localC_)

        cudaFreeAsync(d_localC_, m_stream);
    if (d_localMat_)
        cudaFreeAsync(d_localMat_, m_stream);
    d_localB_ = d_localC_ = d_localMat_ = nullptr;
    if (h_tempB_)
        cudaFreeHost(h_tempB_);
    if (h_tempC_)
        cudaFreeHost(h_tempC_);
    h_tempB_ = h_tempC_ = nullptr;
    // releaseGlobalMats();
}

void GPUWorkspace::allocateLocalMats()
{
    size_t sizeB = maxN_ * m_nRHS * sizeof(cuComplex);
    size_t sizeC = maxM_ * m_nRHS * sizeof(cuComplex);
    size_t sizeMat = maxR_ * m_nRHS * sizeof(cuComplex);
    cudaMallocAsync(&d_localB_, sizeB, m_stream);
    cudaMallocAsync(&d_localC_, sizeC, m_stream);
    cudaMallocAsync(&d_localMat_, sizeMat, m_stream);
    // Allocate pinned host buffers
    cudaMallocHost(&h_tempB_, sizeB);
    cudaMallocHost(&h_tempC_, sizeC);
}

void GPUWorkspace::ensureSize(int M, int N, int R)
{
    if (M > maxM_)
    {
        cuComplex *localC = getLocalC(M);
    }
    if (N > maxN_)
    {
        cuComplex *localB = getLocalB(N);
    }
    if (R > maxR_)
    {
        cuComplex *localMat = getLocalMat(R);
    }
    return;
}

void GPUWorkspace::printMemoryInfo() const
{
    size_t freeMem, totalMem;
    cudaMemGetInfo(&freeMem, &totalMem);
    printf("GPU Memory Info:\n");
    printf("  Total Memory: %zu bytes\n", totalMem);
    printf("  Free Memory: %zu bytes\n", freeMem);
    printf("  Used Memory: %zu bytes\n", totalMem - freeMem);

    printf("  Local B Size: %d bytes\n", maxN_ * m_nRHS * sizeof(cuComplex));
    printf("  Local C Size: %d bytes\n", maxM_ * m_nRHS * sizeof(cuComplex));
    printf("  Local Mat Size: %d bytes\n", maxR_ * m_nRHS * sizeof(cuComplex));
    printf("  Total Local Size: %d bytes\n", maxN_ * m_nRHS * sizeof(cuComplex) + maxM_ * m_nRHS * sizeof(cuComplex) + maxR_ * m_nRHS * sizeof(cuComplex));

    return;
}
