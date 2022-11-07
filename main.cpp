#include "iostream"
#include "pre_process.h"
#include "mpi.h"

#define PERFORMANCE_CHECK true
#define DEBUG true

// TODO: everything concerning MPI

int main(int argc, char *argv[]) {

    /* MAURIZIO */
    std::string filepath = "/home/dmmp/Documents/GitHub/hpc2022/data/gene_expr.tsv"; // TODO: change to CLI args
    // std::string  filepath = "/home/dmmp/Documents/GitHub/hpc2022/data/dummy.csv"; // TODO: change to CLI args

    /* ANTONIO */
    // std::string  filepath = "/Users/azel/Developer/hpc2022/data/gene_expr.tsv"; // TODO: change to CLI args
    // std::string filepath = "/Users/azel/Developer/hpc2022/data/iris.csv"; // TODO: change to CLI args

    //char* file_separator = (char*)("\t");
    char *file_separator = (char *) (",");

    // read mpi?

    // Dataset df = read_data_file(filepath, 79, 2002, 2002, file_separator,".");
    Dataset df = read_data_file(filepath, 100, 5, 5, file_separator, ".", true, false);

    // Initialize the MPI environment
    MPI_Init(nullptr, nullptr);
    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // TODO

    /* Try getting a column */
    std::vector<double> col = df.predictor_matrix.get_col(0);
    std::cout << "\nColumn 0:\n";
    for (int i = 0; i < col.size(); i++) {
        std::cout << col[i] << ", ";
    }

    /* Try getting a row */
    std::vector<double> row = df.predictor_matrix.get_row(0);
    std::cout << "\nRow 0:\n";
    for (int i = 0; i < row.size(); i++) {
        std::cout << row[i] << ", ";
    }


    // Finalize the MPI environment.
    MPI_Finalize();
    return 0; // everything went fine, yay
}
