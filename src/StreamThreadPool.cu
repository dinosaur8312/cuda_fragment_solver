#include "StreamThreadPool.h"
#include <cassert>
#include <chrono>

StreamThreadPool::StreamThreadPool(int num_workers, int nRHS, int matrixSize,const std::vector<GPUTask*>& tasks)
    : num_workers_(num_workers)
{
    streams_.resize(num_workers_);
    threads_.resize(num_workers_);
    ws_.resize(num_workers_);
    solvers_.resize(num_workers_);
    for (int i = 0; i < num_workers_; ++i)
    {
        cudaStreamCreate(&streams_[i]);
        // Initialize GPUWorkspace for each thread
        ws_[i] = new GPUWorkspace(nRHS, matrixSize, streams_[i]);
        solvers_[i] = new GEMMSolver(streams_[i]); // Create a GEMMSolver for each worker
    }
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        for (auto* task : tasks) {
            task_queue_.push(task);
            ++in_flight_tasks_;
        }
    }

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

    for (int i = 0; i < num_workers_; ++i) {
        delete solvers_[i]; // Clean up GEMMSolver if it was created
        delete ws_[i];
        cudaStreamDestroy(streams_[i]);
    }
    std::cout << "StreamThreadPool destructor" << std::endl;
}



void StreamThreadPool::wait() {
    while (in_flight_tasks_ > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}
void StreamThreadPool::workerLoop(int worker_id)
{
    cudaStream_t stream = streams_[worker_id];
    GPUWorkspace *ws = ws_[worker_id];
    GEMMSolver *solver = solvers_[worker_id];
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
            task->setWorkspace(ws_[worker_id]);
            task->setSolver(solver);
            // Print buffer size, stream index, and C++ thread index
            //printf("[Thread %d] Stream %p Buffer sizes: B=%zu, C=%zu, Mat=%zu\n", worker_id, (void *)stream, buffers.sizeB, buffers.sizeC, buffers.sizeMat);
            // Optionally: pass buffer pointers to task if needed
            task->execute(); // Not running actual computation for now
            delete task; // Clean up if task was copied
            in_flight_tasks_.fetch_sub(1);
            printf("Thread %d finished task execution. Tasks in flight: %d\n", worker_id, in_flight_tasks_.load());
        }
    }
    printf("Thread %d finished worker loop.\n", worker_id);
}
