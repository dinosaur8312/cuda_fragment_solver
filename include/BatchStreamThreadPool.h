#pragma once
#include <vector>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <map>
#include <tuple>
#include <iostream>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include "GPUBatchTask.h"
#include "GPUBatchWorkspace.h"
#include "GEMMBatchSolver.h"

class BatchStreamThreadPool {
public:
    BatchStreamThreadPool(int num_workers, int nRHS, int matrixSize, 
                    const std::map<std::tuple<int, int, int>, GPUBatchTask*>& batch_tasks,
                    cuComplex* h_globalMatB);
    ~BatchStreamThreadPool();

   // void enqueueTask(GPUTask* task);
    void wait();

private:
    void workerLoop(int worker_id);

    int num_workers_;
    //std::vector<GPUWorkspace *> ws_; // Each worker has its own workspace
   // std::vector<cudaStream_t> streams_;
    std::vector<std::thread> threads_;
    //std::vector<GEMMSolver *> solvers_; // Each worker has its own GEMMSolver
    std::queue<GPUBatchTask*> task_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::atomic<bool> stop_{false};
    std::atomic<int> in_flight_tasks_{0};

    cuComplex* h_globalMatB_;
    int nRHS_;
    int matrixSize_;
};
