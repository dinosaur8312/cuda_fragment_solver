#include "GPUBatchWorkspace.h"
#include <cuda_runtime_api.h>
#include <cassert>

GPUBatchWorkspace::GPUBatchWorkspace(int nRHS, int matrixSize)
    : m_stream(nullptr), m_nRHS(nRHS), m_matrixSize(matrixSize)
{
    allocateGlobalMats();
}

GPUBatchWorkspace::~GPUBatchWorkspace()
{
    releaseLocalMats();
    releaseGlobalMats();
}

void GPUBatchWorkspace::ensureSize(int M, int N, int R, int tasknum)
{
    // Implement logic to ensure local buffer sizes for batch tasks
    // (similar to GPUWorkspace, but for batched operations)
    // ...
    //cudaDeviceSynchronize(); // Ensure previous operations are completed before resizing
    //fflush(stdout); // Flush stdout to ensure logs are printed before resizing
    //printf("[GPUBatchWorkspace] Ensuring size for tasknum %d: M=%d, N=%d, R=%d\n", tasknum, M, N, R);
    size_t need_LocalBSize = N * m_nRHS * sizeof(cuComplex) * tasknum;
    size_t need_LocalCSize = M * m_nRHS * sizeof(cuComplex) * tasknum;
    size_t need_localMatSize = R * m_nRHS * sizeof(cuComplex) * tasknum;
    size_t need_denseMatSize = M * N * sizeof(cuComplex) * tasknum;
    size_t need_QmatSize = M * R * sizeof(cuComplex) * tasknum;
    size_t need_RmatSize = R * N * sizeof(cuComplex) * tasknum;

    bool taskSizeChanged = false;
    bool LocalBSizeChanged = false;
    bool LocalCSizeChanged = false;
    bool localMatSizeChanged = false;
    bool denseMatSizeChanged = false;
    bool QmatSizeChanged = false;
    bool RmatSizeChanged = false;
    if (tasknum > m_tasknum)
    {
        // If the number of tasks increases, we need to allocate more memory
        m_tasknum = tasknum;
        taskSizeChanged = true;
        //printf("[GPUBatchWorkspace] Task number increased to %d\n", m_tasknum);
        if (d_denseMat_array)
        {
            cudaFreeAsync(d_denseMat_array, m_stream);
            d_denseMat_array = nullptr;
        }
        cudaMallocAsync(&d_denseMat_array, tasknum * sizeof(cuComplex *), m_stream);
        if (d_Qmat_array)
        {
            cudaFreeAsync(d_Qmat_array, m_stream);
            d_Qmat_array = nullptr;
        }
        cudaMallocAsync(&d_Qmat_array, tasknum * sizeof(cuComplex *), m_stream);
        if (d_Rmat_array)
        {
            cudaFreeAsync(d_Rmat_array, m_stream);
            d_Rmat_array = nullptr;
        }
        cudaMallocAsync(&d_Rmat_array, tasknum * sizeof(cuComplex *), m_stream);

        if (d_localB_array)
        {
            cudaFreeAsync(d_localB_array, m_stream);
            d_localB_array = nullptr;
        }
        cudaMallocAsync(&d_localB_array, tasknum * sizeof(cuComplex *), m_stream);
        if (d_localC_array)
        {
            cudaFreeAsync(d_localC_array, m_stream);
            d_localC_array = nullptr;
        }
        cudaMallocAsync(&d_localC_array, tasknum * sizeof(cuComplex *), m_stream);
        if (d_localMat_array)
        {
            cudaFreeAsync(d_localMat_array, m_stream);
            d_localMat_array = nullptr;
        }
        cudaMallocAsync(&d_localMat_array, tasknum * sizeof(cuComplex *), m_stream);
    }

    if (need_LocalBSize > m_localBSize)
    {
        LocalBSizeChanged = true;
        //printf("[GPUBatchWorkspace] LocalB size increased to %zu\n", need_LocalBSize);
        if (d_localB_)
            cudaFreeAsync(d_localB_, m_stream);
        cudaMallocAsync(&d_localB_, need_LocalBSize, m_stream);


        if (h_tempB_)
            cudaFreeHost(h_tempB_);
        cudaMallocHost(&h_tempB_, need_LocalBSize);
        m_localBSize = need_LocalBSize;
    }

    {
        std::vector<cuComplex *> localB_array(tasknum, nullptr);
        for (int i = 0; i < tasknum; ++i)
        {
            localB_array[i] = d_localB_ + i * N * m_nRHS;
        }
        cudaMemcpyAsync(d_localB_array, localB_array.data(), tasknum * sizeof(cuComplex *), cudaMemcpyHostToDevice, m_stream);
    }

    if (need_LocalCSize > m_localCSize)
    {
        LocalCSizeChanged = true;
        if (d_localC_)
            cudaFreeAsync(d_localC_, m_stream);
        cudaMallocAsync(&d_localC_, need_LocalCSize, m_stream);

        if (h_tempC_)
            cudaFreeHost(h_tempC_);
        cudaMallocHost(&h_tempC_, need_LocalCSize);
        m_localCSize = need_LocalCSize;
    }
    {
        std::vector<cuComplex *> localC_array(tasknum, nullptr);
        for (int i = 0; i < tasknum; ++i)
        {
            localC_array[i] = d_localC_ + i * M * m_nRHS;
        }
        cudaMemcpyAsync(d_localC_array, localC_array.data(), tasknum * sizeof(cuComplex *), cudaMemcpyHostToDevice, m_stream);
    }

    if (need_localMatSize > m_localMatSize)
    {
        localMatSizeChanged = true;
        if (d_localMat_)
            cudaFreeAsync(d_localMat_, m_stream);
        cudaMallocAsync(&d_localMat_, need_localMatSize, m_stream);

        m_localMatSize = need_localMatSize;
    }
    //if((localMatSizeChanged)||(taskSizeChanged))
    {
        std::vector<cuComplex *> localMat_array(tasknum, nullptr);
        for (int i = 0; i < tasknum; ++i)
        {
            localMat_array[i] = d_localMat_ + i * R * m_nRHS;
        }
        cudaMemcpyAsync(d_localMat_array, localMat_array.data(), tasknum * sizeof(cuComplex *), cudaMemcpyHostToDevice, m_stream);
    }

    if (R == 0)
    {
        if (need_denseMatSize > m_denseMatSize)
        {
            denseMatSizeChanged = true;
            if (d_denseMat_)
                cudaFreeAsync(d_denseMat_, m_stream);
            cudaMallocAsync(&d_denseMat_, need_denseMatSize, m_stream);

            m_denseMatSize = need_denseMatSize;
        }
       // if(denseMatSizeChanged||taskSizeChanged)
        {
            std::vector<cuComplex *> denseMat_array(tasknum, nullptr);
            for (int i = 0; i < tasknum; ++i)
            {
                denseMat_array[i] = d_denseMat_ + i * M * N;
            }
            cudaMemcpyAsync(d_denseMat_array, denseMat_array.data(), tasknum * sizeof(cuComplex *), cudaMemcpyHostToDevice, m_stream);
        }
    }
    else
    {
        if (need_QmatSize > m_QmatSize)
        {
            QmatSizeChanged = true;
            if (d_Qmat_)
                cudaFreeAsync(d_Qmat_, m_stream);
            cudaMallocAsync(&d_Qmat_, need_QmatSize, m_stream);

            m_QmatSize = need_QmatSize;
        }
       // if (QmatSizeChanged || taskSizeChanged)
        {
            std::vector<cuComplex *> Qmat_array(tasknum, nullptr);
            for (int i = 0; i < tasknum; ++i)
            {
                Qmat_array[i] = d_Qmat_ + i * M * R;
            }
            cudaMemcpyAsync(d_Qmat_array, Qmat_array.data(), tasknum * sizeof(cuComplex *), cudaMemcpyHostToDevice, m_stream);
        }
        

        if (need_RmatSize > m_RmatSize)
        {
            RmatSizeChanged = true;
            if (d_Rmat_)
                cudaFreeAsync(d_Rmat_, m_stream);
            cudaMallocAsync(&d_Rmat_, need_RmatSize, m_stream);

            m_RmatSize = need_RmatSize;
        }
       // if(RmatSizeChanged||taskSizeChanged)
        {
            std::vector<cuComplex *> Rmat_array(tasknum, nullptr);
            for (int i = 0; i < tasknum; ++i)
            {
                Rmat_array[i] = d_Rmat_ + i * R * N;
            }
            cudaMemcpyAsync(d_Rmat_array, Rmat_array.data(), tasknum * sizeof(cuComplex *), cudaMemcpyHostToDevice, m_stream);
        }
    }

   // fflush(stdout); // Ensure logs are printed immediately
    return;
}

size_t GPUBatchWorkspace::getAvailableGPUMemory() const
{
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    return free;
}

void GPUBatchWorkspace::printMemoryInfo() const
{
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    printf("[GPUBatchWorkspace] Free GPU memory: %zu / %zu bytes\n", free, total);
}

void GPUBatchWorkspace::releaseLocalMats()
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

    if (d_denseMat_)
        cudaFreeAsync(d_denseMat_, m_stream);
    if (d_Qmat_)
        cudaFreeAsync(d_Qmat_, m_stream);
    if (d_Rmat_)
        cudaFreeAsync(d_Rmat_, m_stream);
    d_denseMat_ = d_Qmat_ = d_Rmat_ = nullptr;

    if (d_denseMat_array)
    {
        cudaFreeAsync(d_denseMat_array, m_stream);
        d_denseMat_array = nullptr;
    }
    if (d_Qmat_array)
    {
        cudaFreeAsync(d_Qmat_array, m_stream);
        d_Qmat_array = nullptr;
    }
    if (d_Rmat_array)
    {
        cudaFreeAsync(d_Rmat_array, m_stream);
        d_Rmat_array = nullptr;
    }
    if (d_localB_array)
    {
        cudaFreeAsync(d_localB_array, m_stream);
        d_localB_array = nullptr;
    }
    if (d_localC_array)
    {
        cudaFreeAsync(d_localC_array, m_stream);
        d_localC_array = nullptr;
    }
    if (d_localMat_array)
    {
        cudaFreeAsync(d_localMat_array, m_stream);
        d_localMat_array = nullptr;
    }
    m_denseMatSize = 0;
    m_localMatSize = 0;
    m_localBSize = 0;
    m_localCSize = 0;
    m_QmatSize = 0;
    m_RmatSize = 0;
    m_tasknum = 0; // Reset task number
}

void GPUBatchWorkspace::allocateGlobalMats()
{
    if (!h_globalMatC_)
    {
        size_t sizeC = m_matrixSize * m_nRHS * sizeof(cuComplex);
        h_globalMatC_ = (cuComplex *)malloc(sizeC);
    }
}

void GPUBatchWorkspace::releaseGlobalMats()
{
    if (h_globalMatC_)
    {
        free(h_globalMatC_);
        h_globalMatC_ = nullptr;
    }
}
