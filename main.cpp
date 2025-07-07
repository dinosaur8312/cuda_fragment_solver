#include "TaskManager.h"
#include <iostream>
#include <chrono>

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <MIL_input_file>" << std::endl;
        return 1;
    }

    std::string milFile = argv[1];

    // Parameters
    int numStreams = 8;      // Number of CUDA streams to use
    int nRHS = 16;            // Number of right-hand sides per GEMM
    int matrixSize = 2048;   // Size of the global matrix (square, rows × cols)

    try {
        TaskManager manager(numStreams, nRHS, matrixSize);
        manager.loadTasksFromMIL(milFile);
        manager.printTasks();

        //measure time
        auto start = std::chrono::high_resolution_clock::now();

        manager.runAll();
        manager.runAllBatch();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        std::cout << "Total execution time: " << elapsed.count() << " ms" << std::endl;

       // double totalCost = static_cast<double>(manager.getCost()) * nRHS / 1e9; // Convert to GFLOPS
       // std::cout << "Total cost: " << totalCost << " GFLOP" << std::endl;

        double gflops = (static_cast<double>(manager.getCost()) * nRHS) / (elapsed.count() * 1e6); // GFLOPS
        std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;

    } catch (const std::exception &ex) {
        std::cerr << "Exception: " << ex.what() << std::endl;
        return 1;
    }

    std::cout << "All tasks completed." << std::endl;
    return 0;
}
