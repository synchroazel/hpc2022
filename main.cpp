#include "iostream"
#include "pre_process.h"
#include "mpi.h"
#include "limits"
#include <boost/program_options.hpp>


#define PERFORMANCE_CHECK_MAIN true
#define DEBUG_MAIN true
#define CLI_ARGS false
#define PARALLELIZE_INPUT_READ true
#define MASTER_PROCESS 0

// default cli args
#define DEFAULT_CSV_SEPARATOR ","
#define DEFAULT_SKIP_FIRST_ROW false
#define DEFAULT_SKIP_FIRST_COLUMN false

// TODO: everything concerning MPI

int main(int argc, char *argv[]) {
    int MPI_Error_control = 0;
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
            ("rows", po::value<int>(), "csv rows")
            ("columns", po::value<int>(), "csv columns")
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
    //std::string filepath = "/home/dmmp/Documents/GitHub/hpc2022/data/gene_expr.tsv"; // TODO: change to CLI args
     std::string  filepath = "/home/dmmp/Documents/GitHub/hpc2022/data/dummy.csv"; // TODO: change to CLI args

    /* ANTONIO */
    // std::string  filepath = "/Users/azel/Developer/hpc2022/data/gene_expr.tsv"; // TODO: change to CLI args
    // std::string filepath = "/Users/azel/Developer/hpc2022/data/iris.csv"; // TODO: change to CLI args

    //char* file_separator = (char*)("\t");// TODO: change to CLI args
    char *file_separator = (char *) (",");// TODO: change to CLI args

    bool skip_first_row = true;// TODO: change to CLI args
    bool skip_first_column = true;// TODO: change to CLI args

    // int rows = 79;// TODO: change to CLI args
    // int columns = 2002;// TODO: change to CLI args
    // int target_column = 2002;// TODO: change to CLI args

    int rows = 4;// TODO: change to CLI args
    int columns = 3;// TODO: change to CLI args
    int target_column = 3;// TODO: change to CLI args
#endif

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    // Get the rank of the process
    int process_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);



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

    //TODO: search for a better approach
    int rows_per_process, cols_per_process, processes_for_input_read = world_size;
    if(world_size <= rows){
        // just separate the rows
        rows_per_process = (int) ceil(rows/world_size); // es: 79 rows with 8 processes, each process will read up to 10 rows
        cols_per_process = std::numeric_limits<int>::infinity();
    } else if (world_size <= columns){
        // just separate the columns
        rows_per_process = std::numeric_limits<int>::infinity(); // es: 2000 columns with 8 processes, each process will read up to 250 columns
        cols_per_process = (int) ceil(rows/world_size);
    } else if (world_size <= rows * columns){
        // squares of rows and columns
        // TODO: fix
        rows_per_process = (int) ceil(rows/world_size);
        cols_per_process = (int) ceil(rows/world_size);

    } else {
        rows_per_process=1;
        cols_per_process=1;
        processes_for_input_read = rows * columns;
    }

#if DEBUG_MAIN
    if(process_rank == 0){
        std::cout << "There are a total of " << world_size << " processes." << std::endl;
        std::cout << "Each process reads " << rows_per_process << " rows and " << cols_per_process << " columns" << std::endl;
    }
#endif

    Matrix x = Matrix(cols, r); // create matrix
    std::vector<int> y = std::vector<int>(r);

    std::vector<double> local_x = std::vector<double>(r*cols);
    std::vector<int> local_y = std::vector<int>(r);
#if PERFORMANCE_CHECK_MAIN
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    read_dataset_parallel(local_x, local_y, x.m_width, x.r,
                          filepath,
                          process_rank*rows_per_process,
                          process_rank*cols_per_process,
                          rows_per_process, cols_per_process,
                          target_column,
                          file_separator, ".");
    MPI_Error_control = MPI_Allreduce(&local_x, &x.array, r*cols, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if(MPI_Error_control != MPI_SUCCESS){std::cout << "Error during x reduce" << std::endl; exit(1); }
    MPI_Error_control = MPI_Allreduce(&local_y, &y, r, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if(MPI_Error_control != MPI_SUCCESS){std::cout << "Error during y reduce" << std::endl; exit(1); }
#if PERFORMANCE_CHECK_MAIN
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    Dataset df = Dataset(x, y);
#if DEBUG_MAIN
    df.print_dataset(true);
#endif



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
