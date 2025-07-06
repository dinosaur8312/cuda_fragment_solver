cmake_minimum_required(VERSION 3.18)
project(cuda_fragment_solver LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")
# Enable modern CUDA C++
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda -Xcompiler=-Wall")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -g")

# Set default build type to Debug if not specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ flags: ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE_UPPER}}")
message(STATUS "CUDA flags: ${CMAKE_CUDA_FLAGS} ${CMAKE_CUDA_FLAGS_${CMAKE_BUILD_TYPE_UPPER}}")

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Gather source files
file(GLOB_RECURSE SRC_FILES src/*.cu)
file(GLOB MAIN_FILE main.cpp)  # main.cu in project root

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
