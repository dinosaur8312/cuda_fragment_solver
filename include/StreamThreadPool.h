#pragma once
#include <vector>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include "GPUTask.h"
#include "GPUWorkspace.h"
#include "GEMMSolver.h"

class StreamThreadPool {
public:
    StreamThreadPool(int num_workers, int nRHS, int matrixSize, 
                    const std::vector<GPUTask*>& tasks,
                    cuComplex* h_globalMatB);
    ~StreamThreadPool();

   // void enqueueTask(GPUTask* task);
    void wait();

private:
    void workerLoop(int worker_id);

    int num_workers_;
    //std::vector<GPUWorkspace *> ws_; // Each worker has its own workspace
   // std::vector<cudaStream_t> streams_;
    std::vector<std::thread> threads_;
    //std::vector<GEMMSolver *> solvers_; // Each worker has its own GEMMSolver
    std::queue<GPUTask*> task_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::atomic<bool> stop_{false};
    std::atomic<int> in_flight_tasks_{0};

    cuComplex* h_globalMatB_;
    int nRHS_;
    int matrixSize_;
};
