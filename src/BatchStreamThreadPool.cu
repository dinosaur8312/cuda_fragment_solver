#include "BatchStreamThreadPool.h"
#include <cassert>
#include <chrono>
#include "helper_cuda.h"

BatchStreamThreadPool::BatchStreamThreadPool(int num_workers, int nRHS, int matrixSize,
                                   const std::map<std::tuple<int, int, int>, GPUBatchTask*>& batch_tasks,
                                      cuComplex* h_globalMatB) // Added parameter for global matrix B
    : num_workers_(num_workers)
{
    //streams_.resize(num_workers_);
    threads_.resize(num_workers_);
    //ws_.resize(num_workers_);
    //solvers_.resize(num_workers_);
    nRHS_ = nRHS;
    matrixSize_ = matrixSize;

    h_globalMatB_ = h_globalMatB; // Store the global matrix B for use in worker threads

        //std::lock_guard<std::mutex> lock(queue_mutex_);
    for (const auto& pair : batch_tasks)
    {
 //       const auto& dims = pair.first;
        GPUBatchTask* task = pair.second;
        task_queue_.push(task);
        ++in_flight_tasks_;
    }

    // synchronize all threads before starting worker loops
    //cudaDeviceSynchronize();
    //printf("all threads synchronized before starting worker loops\n");

    for (int i = 0; i < num_workers_; ++i)
    {
        threads_[i] = std::thread(&BatchStreamThreadPool::workerLoop, this, i);
    }
}

BatchStreamThreadPool::~BatchStreamThreadPool()
{
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        stop_ = true;
    }
    queue_cv_.notify_all();

    for (auto& t : threads_) {
        if (t.joinable()) t.join();
    }

  //  std::cout << "BatchStreamThreadPool destructor" << std::endl;
}



void BatchStreamThreadPool::wait() {
    while (in_flight_tasks_ > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}





void BatchStreamThreadPool::workerLoop(int worker_id)
{
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    GPUBatchWorkspace *ws = new GPUBatchWorkspace(nRHS_, matrixSize_);
    ws->setGlobalMatB(h_globalMatB_);
    ws->setStream(stream);
    
    GEMMBatchSolver *solver = new GEMMBatchSolver(); // Create a GEMMSolver for each worker
    solver->setStream(stream);
    while (true)
    {
        GPUBatchTask* task = nullptr;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [&]
                            { return stop_ || !task_queue_.empty(); });
            //printf("Thread %d woke up with stop=%d and queue size=%zu\n", worker_id, stop_, task_queue_.size());
            if (stop_ && task_queue_.empty())
                break;
            if (!task_queue_.empty())
            {
                task = task_queue_.front();
                task_queue_.pop();
            }
        }
        if (task)
        {
            task->setStream(stream);
            task->setWorkspace(ws);
            task->setSolver(solver);
            // Print buffer size, stream index, and C++ thread index
            //printf("[Thread %d] Stream %p Buffer sizes: B=%zu, C=%zu, Mat=%zu\n", worker_id, (void *)stream, buffers.sizeB, buffers.sizeC, buffers.sizeMat);
            // Optionally: pass buffer pointers to task if needed
            task->execute(); 
            delete task; // Clean up if task was copied
            in_flight_tasks_.fetch_sub(1);
           // printf("Thread %d finished batch task execution. Batch tasks in flight: %d\n", worker_id, in_flight_tasks_.load());
            //fflush(stdout); // Ensure logs are printed immediately
        }
    }
    printf("Thread %d batch done.\n", worker_id);
    //ws->releaseLocalMats(); // Release any local matrices allocated in the workspace
    delete ws; // Clean up the workspace
    delete solver; // Clean up the GEMMSolver
    cudaStreamDestroy(stream); // Destroy the stream when the worker loop exits
}
