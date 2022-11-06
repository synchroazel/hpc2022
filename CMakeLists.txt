cmake_minimum_required(VERSION 3.23)
project(hpc2022)

set(CMAKE_CXX_STANDARD 14)
# Add MPI Package to Project
find_package(MPI REQUIRED)

add_executable(hpc2022 main.cpp pre_process.h Matrix.h)
# Add libraries for code completion and compiling
target_link_libraries(hpc2022 PUBLIC MPI::MPI_CXX)