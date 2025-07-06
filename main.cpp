#include "TaskManager.h"
#include <iostream>

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <MIL_input_file>" << std::endl;
        return 1;
    }

    std::string milFile = argv[1];

    // Parameters
    int numStreams = 4;      // Number of CUDA streams to use
    int nRHS = 4;            // Number of right-hand sides per GEMM
    int matrixSize = 2048;   // Size of the global matrix (square, rows × cols)

    try {
        TaskManager manager(numStreams, nRHS, matrixSize);
        manager.loadTasksFromMIL(milFile);
        manager.printTasks();
        manager.runAll();
    } catch (const std::exception &ex) {
        std::cerr << "Exception: " << ex.what() << std::endl;
        return 1;
    }

    std::cout << "All tasks completed." << std::endl;
    return 0;
}
