#include "TaskManager.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "utils.h"
#include "StreamThreadPool.h"
TaskManager::TaskManager(int nStreams, int nRHS, int matrixSize)
    : m_nRHS(nRHS), m_matrixSize(matrixSize), m_nStreams(nStreams) {
    // find cuda device
    findCudaDevice(0, nullptr);


}

TaskManager::~TaskManager() {
//    for (auto &stream : m_streams) {
  //      cudaStreamDestroy(stream);
   // }
   std::cout << "Destroying TaskManager with " << m_tasks.size() 
             << " tasks and " << m_batch_tasks.size() 
             << " batch tasks." << std::endl;

}

void TaskManager::addTask(GPUTask* task) {
    m_tasks.push_back(task);
}

void TaskManager::runAll() {
    int max_num_streams = std::thread::hardware_concurrency();
    printf("Max number of streams: %d\n", max_num_streams);
    //
   // int numStreams = 8; // Or use m_tasks.size() or a parameter
    StreamThreadPool pool(m_nStreams, m_nRHS, m_matrixSize, m_tasks);

    pool.wait();
}

void TaskManager::loadTasksFromMIL(const std::string &filePath) {
    auto data = parseMILFile(filePath);
    int id = 0;
    for (const auto &[M, N, R] : data) {
        //printf("Adding task with M: %d, N: %d, R: %d\n", M, N, R);
        auto task = new GPUTask(id++, M, N, R, m_matrixSize, m_nRHS);
        //printf("Task created with ID: %d\n", task->id());
        if (((R==0)&&(M*N<BATCH_DIM_THRESHOLD)) 
        ||(R>0 && (M*R<BATCH_DIM_THRESHOLD) && (N*R<BATCH_DIM_THRESHOLD)) )
        {
            //printf("Adding task to batch_tasks with ID: %d\n", task->id());
            auto key = std::make_tuple(M, N, R);
            auto key_padded = padToNextPowerOf2(key);
            if (m_batch_tasks.find(key_padded) == m_batch_tasks.end()) {
                m_batch_tasks[key_padded] = new GPUBatchTask(key_padded);
            }
            m_batch_tasks[key_padded]->addTask(task);
        } 
        else {
            //printf("Adding task to tasks with ID: %d\n", task->id());
            m_tasks.push_back(task);
        }
    }
}

void TaskManager::printTasks() const {
    for (const auto &task : m_tasks) {
        std::cout << "Single Task ID: " << task->id() 
                  << ", M: " << task->M() 
                  << ", N: " << task->N() 
                  << ", R: " << task->R() 
                  << std::endl;
    }
    // Print batch tasks
    for (const auto &[key, batchTask] : m_batch_tasks) {
        std::cout << "Batch Task with M: " << std::get<0>(key) 
                  << ", N: " << std::get<1>(key)
                  << ", R: " << std::get<2>(key)
                  << ", contains " << batchTask->getTasks().size()
                  << " tasks." << std::endl;
                  /*
        for (const auto &task : batchTask->getTasks()) {
            std::cout << "  Subtask ID: " << task->id()
                      << ", M: " << task->M()
                      << ", N: " << task->N()
                      << ", R: " << task->R()
                      << std::endl;
        }
                      */
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
