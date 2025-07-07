#include "StreamThreadPool.h"
#include <cassert>
#include <chrono>
#include "helper_cuda.h"

StreamThreadPool::StreamThreadPool(int num_workers, int nRHS, int matrixSize,
                                   const std::vector<GPUTask*>& tasks,
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
    for (int i = 0; i < num_workers_; ++i)
    {
        // Initialize GPUWorkspace for each thread
        //ws_[i] = new GPUWorkspace(nRHS, matrixSize);
        //ws_[i]->setGlobalMatB(h_globalMatB); // Set the global matrix B for each workspace
        //solvers_[i] = new GEMMSolver(); // Create a GEMMSolver for each worker
    }
    {
        //std::lock_guard<std::mutex> lock(queue_mutex_);
        for (auto* task : tasks) {
            task_queue_.push(task);
            ++in_flight_tasks_;
        }
    }

    // synchronize all threads before starting worker loops
    //cudaDeviceSynchronize();
    //printf("all threads synchronized before starting worker loops\n");

    for (int i = 0; i < num_workers_; ++i)
    {
        threads_[i] = std::thread(&StreamThreadPool::workerLoop, this, i);
    }
}

StreamThreadPool::~StreamThreadPool()
{
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        stop_ = true;
    }
    queue_cv_.notify_all();

    for (auto& t : threads_) {
        if (t.joinable()) t.join();
    }

    //for (int i = 0; i < num_workers_; ++i) {
      //  delete solvers_[i]; // Clean up GEMMSolver if it was created
       // delete ws_[i];
      //  checkCudaErrors(cudaStreamDestroy(streams_[i])); 
    //}
  //  std::cout << "StreamThreadPool destructor" << std::endl;
}



void StreamThreadPool::wait() {
    while (in_flight_tasks_ > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}
void StreamThreadPool::workerLoop(int worker_id)
{
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    GPUWorkspace *ws = new GPUWorkspace(nRHS_, matrixSize_);
    ws->setGlobalMatB(h_globalMatB_);
    ws->setStream(stream);
    GEMMSolver *solver = new GEMMSolver(); // Create a GEMMSolver for each worker
    solver->setStream(stream);
    while (true)
    {
        GPUTask *task = nullptr;
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
            task->execute(); // Not running actual computation for now
            delete task; // Clean up if task was copied
            in_flight_tasks_.fetch_sub(1);
           // printf("Thread %d finished task execution. Tasks in flight: %d\n", worker_id, in_flight_tasks_.load());
        }
    }
    printf("Thread %d done.\n", worker_id);
    ws->releaseLocalMats(); // Release any local matrices allocated in the workspace
    delete ws; // Clean up the workspace
    delete solver; // Clean up the GEMMSolver
    cudaStreamDestroy(stream); // Destroy the stream when the worker loop exits
}
