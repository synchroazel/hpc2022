cmake_minimum_required(VERSION 3.23)
project(benchmarking C)

set(CMAKE_C_STANDARD 23)
find_package(MPI REQUIRED)

add_executable(benchmarking main.c)
target_link_libraries(benchmarking PUBLIC MPI::MPI_C)