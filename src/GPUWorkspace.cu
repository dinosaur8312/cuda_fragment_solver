#include "GPUWorkspace.h"
#include <cuda_runtime_api.h>
#include "helper_math.h"
#include "helper_cuda.h"

GPUWorkspace::GPUWorkspace(int nRHS, int matrixSize)
    :  m_nRHS(nRHS), m_matrixSize(matrixSize),
      d_localB_(nullptr), d_localC_(nullptr), d_localMat_(nullptr),
      h_tempB_(nullptr), h_tempC_(nullptr), h_globalMatB_(nullptr), h_globalMatC_(nullptr),
      m_denseMatSize(0), m_localMatSize(0), m_localBSize(0), m_localCSize(0),
      m_QmatSize(0), m_RmatSize(0)
{
    allocateGlobalMats();
}

GPUWorkspace::~GPUWorkspace()
{
    //std::cout << "GPUWorkspace::~GPUWorkspace() start" << std::endl;
    //cudaStreamSynchronize(m_stream); // Ensure all operations on the stream are completed before cleanup
    //releaseLocalMats();
    releaseGlobalMats();
    // If you add global matrix management, releaseGlobal();

   // std::cout << "GPUWorkspace::~GPUWorkspace() end" << std::endl;
}

void GPUWorkspace::allocateGlobalMats()
{
    if (!h_globalMatC_)
    {
        size_t sizeC = m_matrixSize * m_nRHS * sizeof(cuComplex);
        h_globalMatC_ = (cuComplex *)malloc(sizeC);
    }
}

void GPUWorkspace::releaseGlobalMats()
{
    //std::cout << "GPUWorkspace::releaseGlobalMats() start" << std::endl;
    if (h_globalMatC_)
    {
        free(h_globalMatC_);
        h_globalMatC_ = nullptr;
    }
   // std::cout << "GPUWorkspace::releaseGlobalMats() end" << std::endl;
}

void GPUWorkspace::releaseLocalMats()
{
    if (d_localB_)
        cudaFreeAsync(d_localB_, m_stream);
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

    if (d_denseMat)
        cudaFreeAsync(d_denseMat, m_stream);
    d_denseMat = nullptr;

    if (d_Qmat)
        cudaFreeAsync(d_Qmat, m_stream);
    d_Qmat = nullptr;

    if (d_Rmat)
        cudaFreeAsync(d_Rmat, m_stream);
    d_Rmat = nullptr;

    m_localBSize = 0;
    m_localCSize = 0;
    m_localMatSize = 0;
    m_denseMatSize = 0;
    m_QmatSize = 0;
    m_RmatSize = 0;
}
/*
__global__ void Qmat_access_kernel(cuComplex *d_Qmat)
{
    // This kernel is just a placeholder for testing
    // It should be replaced with actual logic to initialize or manipulate d_Qmat
    // For now, we just set the first element to a known value for testing
    if (d_Qmat != nullptr)
    {
        d_Qmat[0] = make_cuComplex(1.0f, 0.0f); // Example initialization
        printf("Qmat_access_kernel: d_Qmat[0]= %f + %fi\n", d_Qmat[0].x, d_Qmat[0].y);
    }
}
*/

void GPUWorkspace::ensureSize(int M, int N, int R)
{
   // printf("GPUWorkspace::ensureSize() m_stream %p M = %d, N = %d, R = %d\n", m_stream, M, N, R);
    size_t need_localBSize = N * m_nRHS * sizeof(cuComplex);
    size_t need_localCSize = M * m_nRHS * sizeof(cuComplex);
    size_t need_localMatSize = R * m_nRHS * sizeof(cuComplex);
    size_t need_denseMatSize = M * N * sizeof(cuComplex);
    size_t need_QmatSize = M * R * sizeof(cuComplex);
    size_t need_RmatSize = R * N * sizeof(cuComplex);

    if (need_localBSize > m_localBSize)
    {
   //     printf("GPUWorkspace::ensureSize() need_localBSize = %zu\n", need_localBSize);
        if (d_localB_)
            cudaFreeAsync(d_localB_, m_stream);
        cudaMallocAsync(&d_localB_, need_localBSize, m_stream);

        if (h_tempB_)
            cudaFreeHost(h_tempB_);
        cudaMallocHost(&h_tempB_, need_localBSize);
        m_localBSize = need_localBSize;
       // printf("GPUWorkspace::ensureSize() d_localB_ %p size = %zu\n",
         //      d_localB_, need_localBSize);
        //printf("GPUWorkspace::ensureSize() h_tempB_ %p size = %zu\n",
          //     h_tempB_, need_localBSize);
    }
    if (need_localCSize > m_localCSize)
    {
        //printf("GPUWorkspace::ensureSize() need_localCSize = %zu\n", need_localCSize);
        if (d_localC_)
            cudaFreeAsync(d_localC_, m_stream);
        cudaMallocAsync(&d_localC_, need_localCSize, m_stream);

        if (h_tempC_)
            cudaFreeHost(h_tempC_);
        cudaMallocHost(&h_tempC_, need_localCSize);

        m_localCSize = need_localCSize;
    }
    if (need_localMatSize > m_localMatSize)
    {
        if (d_localMat_)
            cudaFreeAsync(d_localMat_, m_stream);
        cudaMallocAsync(&d_localMat_, need_localMatSize, m_stream);
        m_localMatSize = need_localMatSize;
    }

    if (R == 0)
    {
        if (need_denseMatSize > m_denseMatSize)
        {
          //  printf("GPUWorkspace::ensureSize() need_denseMatSize = %zu\n", need_denseMatSize);
            if (d_denseMat)
                cudaFreeAsync(d_denseMat, m_stream);
            cudaMallocAsync(&d_denseMat, need_denseMatSize, m_stream);
            m_denseMatSize = need_denseMatSize;
        }
    }
    else
    {
        if (need_QmatSize > m_QmatSize)
        {
            if (d_Qmat)
                checkCudaErrors( cudaFreeAsync(d_Qmat, m_stream));
            checkCudaErrors ( cudaMallocAsync(&d_Qmat, need_QmatSize, m_stream));

            m_QmatSize = need_QmatSize;
           // printf("m_stream %p GPUWorkspace::ensureSize() d_Qmat %p size = %zu\n",
             //      m_stream, d_Qmat, need_QmatSize);
        }
        if (need_RmatSize > m_RmatSize)
        {
            if (d_Rmat)
                cudaFreeAsync(d_Rmat, m_stream);
            cudaMallocAsync(&d_Rmat, need_RmatSize, m_stream);
            m_RmatSize = need_RmatSize;
          //  printf("GPUWorkspace::ensureSize() d_Rmat %p size = %zu\n",
                 //  d_Rmat, need_RmatSize);
        }
    }

    //cudaDeviceSynchronize();
    //exit(0);

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

    /*
    printf("  Local B Size: %llu bytes\n", maxN_ * m_nRHS * sizeof(cuComplex));
    printf("  Local C Size: %llu bytes\n", maxM_ * m_nRHS * sizeof(cuComplex));
    printf("  Local Mat Size: %llu bytes\n", maxR_ * m_nRHS * sizeof(cuComplex));
    printf("  Total Local Size: %llu bytes\n", maxN_ * m_nRHS * sizeof(cuComplex) + maxM_ * m_nRHS * sizeof(cuComplex) + maxR_ * m_nRHS * sizeof(cuComplex));
    */
    return;
}
