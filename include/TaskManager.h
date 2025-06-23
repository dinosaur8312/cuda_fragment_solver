#pragma once

#include "GPUTask.h"
#include "GEMMSolver.h"
#include "WorkspaceManager.h"
#include <vector>
#include <memory>
#include <string>
#include <tuple>

class TaskManager {
public:
    TaskManager(int nStreams, int nRHS, int matrixSize);
    ~TaskManager();

    void addTask(std::shared_ptr<GPUTask> task);
    void runAll();
    void loadTasksFromMIL(const std::string& filePath);

private:
    int m_nRHS;
    int m_matrixSize;
    std::vector<cudaStream_t> m_streams;
    std::vector<std::shared_ptr<GPUTask>> m_tasks;

    std::unique_ptr<GEMMSolver> m_solver;
    std::unique_ptr<WorkspaceManager> m_workspace;

    std::vector<std::tuple<int, int, int>> parseMILFile(const std::string &filePath);
};
