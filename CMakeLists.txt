cmake_minimum_required(VERSION 3.23)
project(hpc2022)

set(CMAKE_CXX_STANDARD 14)
# Add MPI Package to Project
find_package(MPI REQUIRED)

add_executable(hpc2022 main.cpp read_dataset.h pre_process.h Dataset.h svm.h tune_svm.h)

# Add libraries for code completion and compiling
target_link_libraries(hpc2022 PUBLIC MPI::MPI_CXX)

# for M1 Macs
include_directories("/opt/homebrew/include")

find_package(Boost 1.55 REQUIRED serialization)
target_link_libraries(hpc2022 PRIVATE Boost::serialization)