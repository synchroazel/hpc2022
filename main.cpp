#include "iostream"
#include "pre_process.h"
#include "mpi.h"

#include <boost/program_options.hpp>


#define PERFORMANCE_CHECK_MAIN true
#define DEBUG_MAIN false
#define CLI_ARGS true
#define PARALLELIZE_INPUT_READ true

// default cli args
#define DEFAULT_CSV_SEPARATOR ","
#define DEFAULT_SKIP_FIRST_ROW false
#define DEFAULT_SKIP_FIRST_COLUMN false

// TODO: everything concerning MPI

int main(int argc, char *argv[]) {
#if CLI_ARGS
    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    namespace po = boost::program_options; // reference: https://www.boost.org/doc/libs/1_80_0/doc/html/program_options/tutorial.html
    // Declare the supported options.
    po::options_description desc("Allowed options");
    desc.add_options()
            ("help", "produce help message")
            ("file_path", po::value<std::string>(), "csv file path")
            ("separator", po::value<std::string>(), "csv file separator")
            //TODO: add others
            ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    //TODO: modify this part
    if (vm.count("help")) {
        cout << desc << "\n";
        return 1;
    }

    if (vm.count("compression")) {
        cout << "Compression level was set to "
             << vm["compression"].as<int>() << ".\n";
    } else {
        cout << "Compression level was not set.\n";
    }

#else
    MPI_Init(nullptr, nullptr);
    /* MAURIZIO */
    std::string filepath = "/home/dmmp/Documents/GitHub/hpc2022/data/gene_expr.tsv"; // TODO: change to CLI args
    // std::string  filepath = "/home/dmmp/Documents/GitHub/hpc2022/data/dummy.csv"; // TODO: change to CLI args

    /* ANTONIO */
    // std::string  filepath = "/Users/azel/Developer/hpc2022/data/gene_expr.tsv"; // TODO: change to CLI args
    // std::string filepath = "/Users/azel/Developer/hpc2022/data/iris.csv"; // TODO: change to CLI args

    char* file_separator = (char*)("\t");// TODO: change to CLI args
    //char *file_separator = (char *) (",");// TODO: change to CLI args

    bool skip_first_row = true;// TODO: change to CLI args
    bool skip_first_column = true;// TODO: change to CLI args

    int rows = 79;// TODO: change to CLI args
    int columns = 2002;// TODO: change to CLI args
    int target_column = 2002;// TODO: change to CLI args
#endif

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);



    // to build matrix
    int cols = columns-1;
    int r = rows;
    if(skip_first_column)cols--;
    if(skip_first_row)r--;



    // Create a new data type called "MPI_Matrix", to represent Matrix Class
    MPI_Datatype MPI_Matrix; // name
    int      count {3}; // 3 elements
    int      block_lengths[3] = { /* lenght of elements*/
            cols * r, /* matrix */
            1, /*m_width*/
            1 /* r */
    };
    MPI_Aint displacements[3]; // array of displacements
    displacements[0] = offsetof (Matrix, array);
    displacements[1] = offsetof (Matrix, m_width);
    displacements[2] = offsetof (Matrix, r);
    MPI_Datatype types[3] = {
            MPI_DOUBLE, /*matrix */
            MPI_INT, /*m_width*/
            MPI_INT /* r */
    };
    MPI_Type_create_struct (count, block_lengths, displacements, types, &MPI_Matrix);
    MPI_Type_commit (&MPI_Matrix);

#if PARALLELIZE_INPUT_READ
    // read mpi

    if(world_rank == 0){
        // create Data structure

        Matrix x = Matrix(cols, r);


    }
#else
     Dataset df = read_data_file_serial(filepath, 79, 2002, 2002, file_separator,".");
    //Dataset df = read_data_file_serial(filepath, 100, 5, 5, file_separator, ".", true, false);
#endif


#if DEBUG_MAIN

    /* Try getting a column */
    std::vector<double> col = df.predictor_matrix.get_col(0);
    std::cout << "\nColumn 0:\n";
    for (double i : col) {
        std::cout << i << ", ";
    }

    /* Try getting a row */
    std::vector<double> row = df.predictor_matrix.get_row(0);
    std::cout << "\nRow 0:\n";
    for (double i : row) {
        std::cout << i << ", ";
    }

#endif
    // Finalize the MPI environment.
    MPI_Finalize();
    return 0; // everything went fine, yay
}
