#include <iostream>
#include "pre_process.h"
#include "mpi.h"

#define PERFORMANCE_CHECK true
#define DEBUG false

// TODO: everything concerning MPI

int main(int argc, char *argv[]) {
     //std::string  filepath = "/home/dmmp/Documents/GitHub/hpc2022/data/gene_expr.tsv"; // TODO: change to CLI args
    std::string  filepath = "/home/dmmp/Documents/GitHub/hpc2022/data/dummy.csv"; // TODO: change to CLI args
    //char* file_separator = (char*)("\t");
    char* file_separator = (char*)(",");

    // read mpi?
    // Dataset df = read_data_file(filepath, 79, 2002, 2002, file_separator,".");
    Dataset df = read_data_file(filepath, 5, 4, 4, file_separator,".");

    // Initialize the MPI environment
    MPI_Init(nullptr, nullptr);
    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // TODO


    // Finalize the MPI environment.
    MPI_Finalize();
    return 0; // everything went fine, yay
}
