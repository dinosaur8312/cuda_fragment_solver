#include "TaskManager.h"
#include <fstream>
#include <sstream>
#include <iostream>

TaskManager::TaskManager(int nStreams, int nRHS, int matrixSize)
    : m_nRHS(nRHS), m_matrixSize(matrixSize) {
    m_solver = std::make_unique<GEMMSolver>();
    m_workspace = std::make_unique<WorkspaceManager>(nRHS);

    m_streams.resize(nStreams);
    for (int i = 0; i < nStreams; ++i) {
        cudaStreamCreate(&m_streams[i]);
    }
}

TaskManager::~TaskManager() {
    for (auto &stream : m_streams) {
        cudaStreamDestroy(stream);
    }
}

void TaskManager::addTask(std::shared_ptr<GPUTask> task) {
    m_tasks.push_back(task);
}

void TaskManager::runAll() {
    size_t streamCount = m_streams.size();
    for (size_t i = 0; i < m_tasks.size(); ++i) {
        auto &task = m_tasks[i];
        cudaStream_t stream = m_streams[i % streamCount];
        task->setStream(stream);
        task->execute(m_solver.get(), *m_workspace, m_nRHS, m_matrixSize);
    }
    cudaDeviceSynchronize();
}

void TaskManager::loadTasksFromMIL(const std::string &filePath) {
    auto data = parseMILFile(filePath);
    int id = 0;
    for (const auto &[M, N, R] : data) {
        auto task = std::make_shared<GPUTask>(id++, M, N, R);
        task->generateRandomMaps();
        task->generateRandomMatrix();
        addTask(task);
    }
}

std::vector<std::tuple<int, int, int>> TaskManager::parseMILFile(const std::string &filePath) {
    std::vector<std::tuple<int, int, int>> parsedData;
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filePath << std::endl;
        return parsedData;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream lineStream(line);
        std::string word;
        std::vector<std::string> words;
        while (lineStream >> word) {
            words.push_back(word);
        }

        if (words.size() >= 7) {
            int M = std::stoi(words[4]);
            int N = std::stoi(words[5]);
            int R = std::stoi(words[6]);
            parsedData.emplace_back(M, N, R);
        }
    }

    file.close();
    return parsedData;
}
