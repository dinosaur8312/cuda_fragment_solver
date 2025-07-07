#include "GPUBatchTask.h"
#include "GPUBatchWorkspace.h"
#include "helper_cuda.h"
#include "GEMMBatchSolver.h"
#include <cstdio>
#include <cstdlib>
#include <cassert>

GPUBatchTask::GPUBatchTask(int M_pad, int N_pad, int R_pad)
    : m_M_pad(M_pad), m_N_pad(N_pad), m_R_pad(R_pad) {
}

GPUBatchTask::~GPUBatchTask() {
    // Clean up tasks
    for (auto task : m_tasks) {
        delete task;
    }
    m_tasks.clear();

}

void GPUBatchTask::addTask(GPUTask* task) {
    if(m_tasks.size()==0) {
        // Set the matrix size based on the first task
        m_matrixSize = task->maxtrixSize();
        m_nRHS = task->nRHS();
    } else {
        // Check if the new task has the same dimensions as existing tasks
        if (task->maxtrixSize() != m_matrixSize || task->nRHS() != m_nRHS) {
            fprintf(stderr, "Error: All tasks in a batch must have the same matrix size and number of RHS.\n");
            exit(EXIT_FAILURE);
        }
    }
    m_tasks.push_back(task);
}

void GPUBatchTask::uploadBatchToDevice() {
    // Upload all tasks to the device
    for (auto task : m_tasks) {
        task->uploadToDevice();
    }
}

void GPUBatchTask::generateRandomMaps(int matrixSize) {
    // Generate random source and sink maps for each task
    for (auto task : m_tasks) {
        task->generateRandomMaps(matrixSize);
    }
}

void GPUBatchTask::generateRandomMatrices()
{
    //set all value to one for debugging
    if (m_R_pad == 0) // Dense matrix case
    {
        m_denseMat.resize(m_M_pad * m_N_pad* m_tasks.size());
        for (auto &val : m_denseMat)
            val = make_cuComplex(1.0f, 0.0f);
            //val = make_cuComplex(static_cast<float>(rand()) / RAND_MAX, static_cast<float>(rand()) / RAND_MAX);
    }
    else
    {
        m_Qmat.resize(m_M_pad * m_R_pad * m_tasks.size());
        m_Rmat.resize(m_R_pad * m_N_pad * m_tasks.size());
        for (auto &val : m_Qmat)
            val = make_cuComplex(1.0f, 0.0f);
        for (auto &val : m_Rmat)
            val = make_cuComplex(1.0f, 0.0f);
    }
}

void GPUBatchTask::uploadToDevice()
{
    if (m_R_pad == 0) // Dense matrix case
    {
        size_t sizeDenceMat = m_M_pad * m_N_pad* sizeof(cuComplex) * m_tasks.size();
        checkCudaErrors(cudaMemcpyAsync(d_denseMat_, m_denseMat.data(), sizeDenceMat, cudaMemcpyDefault, m_stream));
    }
    else
    {
        size_t sizeQmat = m_M_pad * m_R_pad * sizeof(cuComplex) * m_tasks.size();
        size_t sizeRmat = m_R_pad * m_N_pad * sizeof(cuComplex) * m_tasks.size();

        checkCudaErrors( cudaMemcpyAsync(d_Qmat_, m_Qmat.data(), sizeQmat, cudaMemcpyHostToDevice, m_stream));
        checkCudaErrors( cudaMemcpyAsync(d_Rmat_, m_Rmat.data(), sizeRmat,cudaMemcpyHostToDevice, m_stream));
    }
}

void GPUBatchTask::gather()
{
    for(int itask=0; itask<(int)(m_tasks.size()); itask++)
    {
        int N = m_tasks[itask]->N();
        cuComplex *h_tempB = h_tempB_ + itask * m_N_pad*m_nRHS; // Pinned memory for temporary B matrix
        const std::vector<int> &srcMap = m_tasks[itask]->getSrcMap();
        for(int j=0; j<m_nRHS; j++)
        {
            for(int i=0; i<N; i++)
            {
                // Gather from global matrix B to temporary pinned memory
                h_tempB[j * m_N_pad + i] = h_globalMatB_[srcMap[i] + j * m_matrixSize];
            }
        }
    }
    size_t gatherSize = sizeof(cuComplex) * m_nRHS * m_N_pad * m_tasks.size();
    cudaMemcpyAsync(d_localB_, h_tempB_, gatherSize, cudaMemcpyHostToDevice, m_stream);

}

void GPUBatchTask::scatter()
{
    cudaMemcpyAsync(h_tempC_, d_localC_, m_nRHS * m_M_pad * m_tasks.size() * sizeof(cuComplex), cudaMemcpyDeviceToHost, m_stream);

    for(int itask=0; itask<(int)(m_tasks.size()); itask++)
    {
        int M = m_tasks[itask]->M();
        cuComplex *h_tempC = h_tempC_ + itask * m_M_pad*m_nRHS; // Pinned memory for temporary C matrix
        const std::vector<int> &sinkMap = m_tasks[itask]->getSinkMap();
        for(int j=0; j<m_nRHS; j++)
        {
            for(int i=0; i<M; i++)
            {
                // Scatter from temporary pinned memory to global matrix C
                h_globalMatC_[sinkMap[i] + j * m_matrixSize] += h_tempC[j * m_M_pad + i];
            }
        }
    }
}

void GPUBatchTask::execute()
{
   // generateRandomMaps(m_matrixSize);
    for (auto task : m_tasks) {
        //task->setStream(m_stream);
        //printf("Executing task %d\n", task->id());
        task->generateRandomMaps(m_matrixSize);
    }
    generateRandomMatrices();
    uploadToDevice();
    gather();
    solve();
    scatter();


    return;
}

void GPUBatchTask::setWorkspace(GPUBatchWorkspace* workspace) {
    m_workspace = workspace;
    m_workspace->ensureSize(m_M_pad, m_N_pad, m_R_pad, m_tasks.size());

    h_tempB_ = m_workspace->getPinnedTempB();
    h_tempC_ = m_workspace->getPinnedTempC();
    d_localB_ = m_workspace->getLocalB();
    d_localC_ = m_workspace->getLocalC();
    d_localMat_ = m_workspace->getLocalMat();
    d_denseMat_ = m_workspace->getDenseMat();
    h_globalMatB_ = m_workspace->getGlobalMatB();
    h_globalMatC_ = m_workspace->getGlobalMatC();
    d_Qmat_ = m_workspace->getQmat();
    d_Rmat_ = m_workspace->getRmat();

    d_denseMat_array = m_workspace->getDenseMatArray();
    d_Qmat_array = m_workspace->getQmatArray();
    d_Rmat_array = m_workspace->getRmatArray();
    d_localB_array = m_workspace->getLocalBArray();
    d_localC_array = m_workspace->getLocalCArray();
    d_localMat_array = m_workspace->getLocalMatArray();


}


void GPUBatchTask::solve()
{
    assert(m_solver != nullptr && m_workspace != nullptr);
    //GPUWorkspace *ws = m_workspace;
    GEMMBatchSolver *solver = m_solver;

    if (m_R_pad == 0)
    {
       //printf("Single GEMM\n");
        solver->gemm(d_denseMat_array, d_localB_array, d_localC_array, m_M_pad, m_N_pad, m_nRHS, m_tasks.size());
    }
    else
    {
     //   printf("B2B GEMM\n");
        solver->gemm(d_Rmat_array, d_localB_array, d_localMat_array, m_R_pad, m_N_pad, m_nRHS, m_tasks.size());
        solver->gemm(d_Qmat_array, d_localMat_array, d_localC_array, m_M_pad, m_R_pad, m_nRHS, m_tasks.size());
    }

    //cudaDeviceSynchronize();
    //exit(0);
}