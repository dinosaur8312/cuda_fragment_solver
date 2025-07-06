#include "GPUBatchTask.h"

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
    m_tasks.push_back(task);
}

void GPUBatchTask::uploadBatchToDevice() {
    // Upload all tasks to the device
    for (auto task : m_tasks) {
        task->uploadToDevice();
    }
}