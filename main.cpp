#include "iostream"
#include "pre_process.h"
#include "mpi.h"  // vanilla MPI
#include "limits"
#include "math.h"
#include <boost/program_options.hpp>

#include "svm.hpp"
#include "svm.cpp"

#define PERFORMANCE_CHECK_MAIN true
#define DEBUG_MAIN true
#define DEBUG_SVM true
#define CLI_ARGS false
#define PARALLELIZE_INPUT_READ true
#define MASTER_PROCESS 0

// default cli args
#define DEFAULT_CSV_SEPARATOR ","
#define DEFAULT_SKIP_FIRST_ROW false
#define DEFAULT_SKIP_FIRST_COLUMN false

// function prototypes
 void Set_Kernel(std::string ker_type, KernelFunc &K, std::vector<double> &params);

// TODO: implement CLI args

void build_mpi_datatype(MPI_Datatype* MPI_Dataset, Dataset df ){
    // Create a new data type called "MPI_Matrix", to represent Matrix Class

    int      count {6}; // 6 elements
    int      block_lengths[6] = { /* lenght of elements*/
            (int)(df.rows_number*df.predictors_column_number), /*predictor matrix*/
            (int)(df.rows_number),/*class vector*/
            1, /*predictors_column_number*/
            1, /* rows_number */
            (int)(df.number_of_unique_classes), /*unique classes*/
            1 /*number of unique classes*/
    };
    MPI_Aint displacements[6]; // array of displacements
    displacements[0] = offsetof (Dataset, predictor_matrix);
    displacements[1] = offsetof (Dataset, class_vector);
    displacements[2] = offsetof (Dataset, predictors_column_number);
    displacements[3] = offsetof (Dataset, rows_number);
    displacements[4] = offsetof (Dataset, unique_classes);
    displacements[5] = offsetof (Dataset, number_of_unique_classes);
    MPI_Datatype types[6] = {
            MPI_DOUBLE, /*predictor matrix */
            MPI_INT, /*class array*/
            MPI_INT, /*predictors_column_number*/
            MPI_INT, /* rows_number */
            MPI_INT, /*unique classes*/
            MPI_INT /*number of unique classes*/
    };
    MPI_Type_create_struct (count, block_lengths, displacements, types, MPI_Dataset);
    MPI_Type_commit (MPI_Dataset);
}


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
    //std::string filepath = "/home/dmmp/Documents/GitHub/hpc2022/data/gene_expr.csv"; // TODO: change to CLI args
    //std::string filepath = "/home/dmmp/Documents/GitHub/hpc2022/data/dummy.csv"; // TODO: change to CLI args

    /* ANTONIO */
    // std::string filepath = "/Users/azel/Developer/hpc2022/data/dummy.csv"; // TODO: change to CLI args
    std::string filepath = "/Users/azel/Developer/hpc2022/data/iris_train.csv"; // TODO: change to CLI args
    // std::string filepath = "/Users/azel/Developer/hpc2022/data/gene_expr.csv"; // TODO: change to CLI args


    char *file_separator = (char *) (",");// TODO: change to CLI args

    int rows = 69;// TODO: change to CLI args
    int columns = 5;// TODO: change to CLI args
    int target_column = 5;// TODO: change to CLI args

   // int rows = 4;// TODO: change to CLI args
   // int columns = 3;// TODO: change to CLI args
   // int target_column = 3;// TODO: change to CLI args
#endif

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    // Get the rank of the process
    int process_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);



    // to build matrix
    int cols = columns-1; // because of the y column
    int r = rows;


#if PARALLELIZE_INPUT_READ
    // read mpi

    //TODO: search for a better approach

    // reading techniques
    int rows_per_process, cols_per_process, processes_for_input_read = world_size;
    if(world_size <= rows){
        // just separate the rows
        rows_per_process = (int) std::ceil((double) rows/ (double) world_size); // es: 79 rows with 8 processes, each process will read up to 10 rows
        cols_per_process = 0;
#if DEBUG_MAIN
        if(process_rank == MASTER_PROCESS){
            std::cout << "Case 1: \nEach process reads up to " << rows_per_process << " rows and all columns" << std::endl;
        }

#endif
    } else if (world_size <= columns){
        // just separate the columns
        rows_per_process = 0; // es: 2000 columns with 8 processes, each process will read up to 250 columns
        cols_per_process = (int) std::ceil(rows/world_size);
#if DEBUG_MAIN
        if(process_rank == MASTER_PROCESS){
            std::cout << "Case 2: \nEach process reads all rows and up to " << cols << " columns" << std::endl;
        }


#endif
    } else if (world_size <= rows * columns){
        // squares of rows and columns
        // TODO: find a better way
        if(rows > columns){
            rows_per_process = rows;
            cols_per_process = 0;
            processes_for_input_read = rows;
        } else {
            rows_per_process = 0;
            cols_per_process = columns;
            processes_for_input_read = columns;
        }


#if DEBUG_MAIN
        if(process_rank == MASTER_PROCESS){
            std::cout << "Case 3: \nEach process reads up to " << rows_per_process << " rows and "<< cols_per_process << " columns" << std::endl;
        }

#endif
    } else {
        rows_per_process=1;
        cols_per_process=1;
        processes_for_input_read = rows * columns;
#if DEBUG_MAIN
        if(process_rank == MASTER_PROCESS){
            std::cout << "Case 4: \nEach process reads element. You should consider linear read" << std::endl;
        }

#endif
    }

#if DEBUG_MAIN
    if(process_rank == MASTER_PROCESS){
        std::cout << "There are a total of " << world_size << " processes." << std::endl;
    }
#endif


    auto* final_x = (double*) calloc(rows*columns,sizeof(double ));
    auto* y = (int*) calloc(rows, sizeof (int));

    auto* local_x = (double*) calloc(rows*columns,sizeof(double ));
    auto* local_y = (int*) calloc(rows, sizeof (int));
#if DEBUG_MAIN
    std::cout << "\n\nProcess rank: " << process_rank << std::endl <<
        "I will read from row " << process_rank * rows_per_process << " to " << process_rank * rows_per_process + rows_per_process - 1
        << std::endl << "and from column " << process_rank * cols_per_process << " to column " << process_rank * cols_per_process + cols_per_process - 1 << "\n\n";
#endif
#if PERFORMANCE_CHECK_MAIN
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    if(process_rank <= processes_for_input_read ) {
        read_dataset_parallel(local_x, local_y, cols, r,
                              filepath,
                              process_rank * rows_per_process,
                              process_rank * cols_per_process,
                              rows_per_process, cols_per_process,
                              target_column,
                              file_separator);
    }
#if DEBUG_READ_DATA
    // std::cout << "final_x before reduce: " << std::endl;
    // print_matrix(final_x, r, cols, true);
//
    // std::cout << "local x before reduce: " << std::endl;
    // print_matrix(local_x, r, cols, true);
//
    // std::cout << "count for reduce: " << r*cols << std::endl;
//
//
    // std::cout << "BEFORE:\n"<< std::endl;
    // std::cout << "r:\n" << r << std::endl;
    // std::cout << "cols:\n" << cols << std::endl;
//
    // int size = r*cols;
    // std::cout << "size:\n" << size << std::endl;

#endif

    MPI_Error_control = MPI_Allreduce(local_x, final_x, r*cols, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if(MPI_Error_control != MPI_SUCCESS){std::cout << "Error during x reduce" << std::endl; exit(1); }
#if DEBUG_READ_DATA
    if(process_rank == MASTER_PROCESS){
        std::cout << "AFTER:\n"<< std::endl;
        std::cout << "r:\n" << r << std::endl;
        std::cout << "cols:\n" << cols << std::endl;


        std::cout << "X after reduce: " << std::endl;
        print_matrix(final_x, r, cols, true);
    }

#endif
    MPI_Error_control = MPI_Allreduce(local_y, y, r, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if(MPI_Error_control != MPI_SUCCESS){std::cout << "Error during y reduce" << std::endl; exit(1); }
#if DEBUG_READ_DATA
    if(process_rank == MASTER_PROCESS){
        std::cout << "Y after reduce: " << std::endl;
        print_vector(y,rows);
        std::cout << std::endl;
    }

#endif
#if PERFORMANCE_CHECK_MAIN
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    free(local_x);
    free(local_y);

    Dataset df;
    df.rows_number = r;
    df.predictor_matrix = final_x;
    df.predictors_column_number = cols;
    df.class_vector = y;
    df.number_of_unique_classes = get_number_of_unique_classes(df.class_vector,df.rows_number);
    df.unique_classes = (int*) calloc(df.number_of_unique_classes, sizeof(int));
    get_unique_classes(df.class_vector, df.rows_number, df.number_of_unique_classes, df.unique_classes);


#if DEBUG_MAIN
    if(process_rank == MASTER_PROCESS) print_dataset(df, false);
#endif



#else
     Dataset df = read_data_file_serial(filepath, 79, 2002, 2002, file_separator,".");
    //Dataset df = read_data_file_serial(filepath, 100, 5, 5, file_separator, ".", true, false);
#endif


#if DEBUG_MAIN


    MPI_Barrier(MPI_COMM_WORLD);

    if(process_rank == MASTER_PROCESS){
        /* Try getting a column */
        auto* col = (double *) calloc(df.rows_number, sizeof (double ));

        //std::cout << "Before get column:" << std::endl;
        //print_vector(col,df.rows_number);

        get_column(df,0, col);
        std::cout << "\nColumn 0:\n";
        print_vector(col,df.rows_number);
        free(col);

        std::cout << "Column has been freed. Trying row now:" << std::endl;

        /* Try getting a row */
        auto* row = (double *) calloc(df.predictors_column_number, sizeof (double ));

        //std::cout << "Before get row:" << std::endl;
        //print_vector(row,df.predictors_column_number);

        get_row(df, 0, false,row);
        std::cout << "\nRow 0:\n";
        print_vector(row,df.predictors_column_number);
        free(row);
        std::cout << "Row has been freed." << std::endl;

#if DEBUG_SVM

        /* Try SVM SERIAL IMPLEMENTATION */

        std::string ker_type = "rbf";

        KernelFunc K;
        std::vector<double> params;
        Set_Kernel(ker_type, K, params);

        size_t D = 0;
        double C = 0.1;
        double lr = 0.0001;

        Kernel_SVM svm(K, params, true);
        svm.train(df, D, C, lr, 0.001);

        // svm.test(df_test);


#endif

    }



#endif
    // Finalize the MPI environment.
    MPI_Finalize();
    free(final_x);
    free(y);

    return 0; // everything went fine, yay
}


// if Kernel is polynomial
double c = 0.1;  // TODO: change to CLI args
double d;  // TODO: change to CLI args

// if Kernel is rbf
double gamma = 1;  // TODO: change to CLI args

void Set_Kernel(std::string ker_type, KernelFunc &K, std::vector<double> &params) {

    if (ker_type == "linear") {
        K = kernel::linear;
    } else if (ker_type == "polynomial") {
        K = kernel::polynomial;
        params = {c, d};
    } else if (ker_type == "rbf") {
        K = kernel::rbf;
        params = {gamma};
    }
}