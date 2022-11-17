#include "iostream"
#include "pre_process.h"
#include "mpi.h"

#include <cmath>
//#include "math.h"

#define MASTER_PROCESS 0
#define DEBUG_READ_DATASET false
#define PERFORMANCE_CHECK true
void build_mpi_datatype(MPI_Datatype *MPI_Dataset, Dataset df) {

    /**
     * Create a new data type called "MPI_Matrix", to represent Matrix Class
     */

    int count{6}; // 6 elements
    int block_lengths[6] = { /* length of elements*/
            (int) (df.rows_number * df.predictors_column_number), /*predictor matrix*/
            (int) (df.rows_number),/*class vector*/
            1, /*predictors_column_number*/
            1, /* rows_number */
            (int) (df.number_of_unique_classes), /*unique classes*/
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
    MPI_Type_create_struct(count, block_lengths, displacements, types, MPI_Dataset);
    MPI_Type_commit(MPI_Dataset);
}

Dataset read_dataset(const std::string& filepath, int rows, int columns, int target_column) {

    /**
     * Read dataset from a file, given filepath, rows, columns and target column
     */

    int MPI_Error_control = 0;

    char *file_separator = (char *) (",");

    // TODO : pass rank and size to function

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    // Get the rank of the process
    int process_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);

    // to build matrix
    int cols = columns - 1; // because of the y column
    int r = rows;


    // reading techniques
    int rows_per_process, cols_per_process, processes_for_input_read = world_size;
    if (world_size <= rows) {
        // just separate the rows
        rows_per_process = (int) std::ceil((double) rows /
                                           (double) world_size); // es: 79 rows with 8 processes, each process will read up to 10 rows
        cols_per_process = 0;

        if (process_rank == MASTER_PROCESS) {
            std::cout << "Case 1: \nEach process reads up to " << rows_per_process << " rows and all columns"
                      << std::endl;
        }

    } else if (world_size <= columns) {
        // just separate the columns
        rows_per_process = 0; // es: 2000 columns with 8 processes, each process will read up to 250 columns
        cols_per_process = (int) std::ceil(rows / world_size);

        if (process_rank == MASTER_PROCESS) {
            std::cout << "Case 2: \nEach process reads all rows and up to " << cols << " columns" << std::endl;
        }

    } else if (world_size <= rows * columns) {
        // squares of rows and columns
        // TODO: find a better way
        if (rows > columns) {
            rows_per_process = rows;
            cols_per_process = 0;
            processes_for_input_read = rows;
        } else {
            rows_per_process = 0;
            cols_per_process = columns;
            processes_for_input_read = columns;
        }

        if (process_rank == MASTER_PROCESS) {
            std::cout << "Case 3: \nEach process reads up to " << rows_per_process << " rows and " << cols_per_process
                      << " columns" << std::endl;
        }

    } else {
        rows_per_process = 1;
        cols_per_process = 1;
        processes_for_input_read = rows * columns;

        if (process_rank == MASTER_PROCESS) {
            std::cout << "Case 4: \nEach process reads element. You should consider linear read" << std::endl;
        }
    }

    if (process_rank == MASTER_PROCESS) {
        std::cout << "There are a total of " << world_size << " processes." << std::endl;
    }

    auto *final_x = (double *) calloc(rows * columns, sizeof(double));
    auto *y = (int *) calloc(rows, sizeof(int));

    auto *local_x = (double *) calloc(rows * columns, sizeof(double));
    auto *local_y = (int *) calloc(rows, sizeof(int));
#if DEBUG_READ_DATASET
    std::cout << "\n\nProcess rank: " << process_rank << std::endl <<
              "I will read from row " << process_rank * rows_per_process << " to "
              << process_rank * rows_per_process + rows_per_process - 1
              << std::endl << "and from column " << process_rank * cols_per_process << " to column "
              << process_rank * cols_per_process + cols_per_process - 1 << "\n\n";
#endif
#if PERFORMANCE_CHECK
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    if (process_rank <= processes_for_input_read) {

        read_dataset_parallel(local_x,
                              local_y,
                              cols,
                              r,
                              filepath,
                              process_rank * rows_per_process,
                              process_rank * cols_per_process,
                              rows_per_process,
                              cols_per_process,
                              target_column,
                              file_separator);

    }

    MPI_Error_control = MPI_Allreduce(local_x, final_x, r * cols, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if (MPI_Error_control != MPI_SUCCESS) {
        std::cout << "Error during x reduce" << std::endl;
        exit(1);
    }

    MPI_Error_control = MPI_Allreduce(local_y, y, r, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (MPI_Error_control != MPI_SUCCESS) {
        std::cout << "Error during y reduce" << std::endl;
        exit(1);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    free(local_x);
    free(local_y);

    Dataset df;
    df.rows_number = r;
    df.predictor_matrix = final_x;
    df.predictors_column_number = cols;
    df.class_vector = y;
    df.number_of_unique_classes = get_number_of_unique_classes(df.class_vector, df.rows_number);
    df.unique_classes = (int *) calloc(df.number_of_unique_classes, sizeof(int));
    get_unique_classes(df.class_vector, df.rows_number, df.unique_classes);
#if PERFORMANCE_CHECK
    MPI_Barrier(MPI_COMM_WORLD);
#endif
#if DEBUG_READ_DATASET
    if (process_rank == MASTER_PROCESS) print_dataset(df, true);
#endif
    return df;

}