cmake_minimum_required(VERSION 3.18)
project(cuda_fragment_solver LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Enable modern CUDA C++
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda -Xcompiler=-Wall")

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Gather source files
file(GLOB_RECURSE SRC_FILES src/*.cu)
file(GLOB MAIN_FILE main.cu)  # main.cu in project root

# Add executable
add_executable(cuda_fragment_solver ${MAIN_FILE} ${SRC_FILES})

# Link cuBLAS
find_package(CUDAToolkit REQUIRED)
target_link_libraries(cuda_fragment_solver PRIVATE CUDA::cublas)

# Optional: enable Thrust CCCL compatibility
target_compile_definitions(cuda_fragment_solver PRIVATE THRUST_IGNORE_DEPRECATED_CPP_DIALECT=1)

# Set output directory
set_target_properties(cuda_fragment_solver PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
