#pragma once

#include "GPUTask.h"
#include "GPUBatchTask.h"
#include "GEMMSolver.h"
#include "GPUWorkspace.h"
#include <vector>
#include <memory>
#include <string>
#include <tuple>
#include <map>

#define BATCH_DIM_THRESHOLD  16384 // Threshold for batch dimension, adjust as needed;


class TaskManager {
public:
    TaskManager(int nStreams, int nRHS, int matrixSize);
    ~TaskManager();

    void addTask(GPUTask* task);
    void printTasks() const;
    void runAll();
    void loadTasksFromMIL(const std::string& filePath);

private:
    int m_nRHS;
    int m_matrixSize;
    //std::vector<cudaStream_t> m_streams;
    std::vector<GPUTask *> m_tasks;
    //std::vector<GPUBatchTask *> m_batch_tasks;
    std::map <std::tuple<int, int, int>, GPUBatchTask *> m_batch_tasks;
    int m_nStreams;
    cuComplex *h_globalMatB_ = nullptr; // Host global matrix B shared across all threads

    

    //std::unique_ptr<GEMMSolver> m_solver;
    //std::unique_ptr<GPUWorkspace> m_workspace;

    std::vector<std::tuple<int, int, int>> parseMILFile(const std::string &filePath);
};
