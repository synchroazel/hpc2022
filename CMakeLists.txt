cmake_minimum_required(VERSION 3.15)
project(hpc2022)

set(CMAKE_CXX_STANDARD 14)

# Add MPI Package to Project
find_package(MPI REQUIRED)

add_executable(hpc2022 main.cpp read_dataset.h pre_process.h Dataset.h svm.h tune_svm.h)

# Add libraries for code completion and compiling
target_link_libraries(hpc2022 PUBLIC MPI::MPI_CXX)

# for homebrew-installed libraries
include_directories("/opt/homebrew/include")
include_directories("/opt/homebrew/Cellar/libomp/15.0.5/include")

# for boost libraries inside they cluster
include_directories("~/include")

add_definitions(-DBOOST_ERROR_CODE_HEADER_ONLY)