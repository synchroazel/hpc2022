#include "iostream"
#include "mpi.h"
#include <ctime>
#include <sys/time.h>

#include "Dataset.h"
#include "tune_svm.h"
#include "read_dataset.h"


/* Macro to switch modes */
#define CLI_ARGS false
#define DEBUG_MPI_OPERATIONS false
#define SHOW_LOGTIME true
#define IMPLEMENTED_KERNELS 4
#define NUMBER_OF_HYPER_PARAMETERS 4
#define NUMBER_OF_PERFORMANCE_CHECKS 15


void logtime() {

    int process_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);

    char buffer[26];
    int millisec;
    struct tm *tm_info;
    struct timeval tv;

    gettimeofday(&tv, NULL);

    millisec = lrint(tv.tv_usec / 1000.0); // Round to nearest millisec
    if (millisec >= 1000) { // Allow for rounding up to nearest second
        millisec -= 1000;
        tv.tv_sec++;
    }

    tm_info = localtime(&tv.tv_sec);

    strftime(buffer, 26, "%Y-%m-%d %H:%M:%S", tm_info);

#if SHOW_LOGTIME
    printf("[rank %d at %s.%03d] ", process_rank, buffer, millisec);
#endif

}


enum train_flag {
    training = 0, testing = 1, tuning = 2
} flag;



//  TODO:
//        implement:
//                  cli args
//                  open_mp
//                  mpi logic for train and test
//                  radial tuning
//                  sigmoid tuning
//                  poly tuning
//                  all together logic
//        debug:
//                  svm test



int main(int argc, char *argv[]) {

    /* Initialize MPI ------------------------------------ */

    MPI_Init(nullptr, nullptr);  // TODO : check argc & argv

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    // Get the rank of the process
    int process_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);

    /* --------------------------------------------------- */

    int control;

#if PERFORMANCE_CHECK

    MPI_Barrier(MPI_COMM_WORLD);

    if (process_rank == MASTER_PROCESS) {
        logtime();
        std::cout << "This step is done by process 0 only to benchmark the speed of for loops\n";
        double start = MPI_Wtime(), end = 0;
        for (int i = 0; i < 1000; i++) {}
        end = MPI_Wtime();
        logtime();
        std::cout << "On this platform, a for loop cycle alone takes an average of " << (end - start) / 1000 << "\n\n";

    }
    MPI_Barrier(MPI_COMM_WORLD);

#endif

    // enum relevant_metric {Accuracy = NUMBER_OF_HYPER_PARAMETERS + 1, AccuracyC1 = NUMBER_OF_HYPER_PARAMETERS +2, AccuracyC3 = NUMBER_OF_HYPER_PARAMETERS +3}metric;

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

    /**
     * Flag selection (training, testing, tuning)
     */

    flag = testing;


    // std::string filepath_training = "/home/dmmp/Documents/GitHub/hpc2022/data/iris_train.csv";
    std::string filepath_training = "/Users/azel/Developer/hpc2022/data/iris_train.csv";
    std::string filepath_validation = "../data/iris_validation.csv";
    std::string filepath_hyper_parameters = "../data/hyperparameters.csv"; // TODO: implement
    std::string filepath_svm = "../saved_svm/radialr_C0.100000_G1.000000.dat";
    size_t rows_t = 70, rows_v = 30, columns = 5, target_column = 5;
    char ker_type = 'l';

    bool verbose = true;

    std::string save_dir_path = "/Users/azel/Developer/hpc2022/saved_svm/";

    double Cost = 5;
    double gamma = 0.1;
    double coef0 = 0;
    double degree = 1;

    const double lr = 0.0001;
    const double limit = 0.1;

    const double eps = DEFAULT_EPS;

#endif

#if PERFORMANCE_CHECK


    /**
     * Program startup
     */

    auto *time_checks = (double *) calloc(NUMBER_OF_PERFORMANCE_CHECKS, sizeof(double));
    int time_iterator = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    *(time_checks + time_iterator) = MPI_Wtime(); // start

    if (process_rank == MASTER_PROCESS) {
        logtime();
        std::cout << "Program starts at time " << *(time_checks + time_iterator) << "\n" << std::endl;
    }
    ++time_iterator;

#endif

    /**
     * Switch start
     */


    switch (flag) {


        /// Training case

        case train_flag::training : {

// -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#if PERFORMANCE_CHECK

            MPI_Barrier(MPI_COMM_WORLD);
            *(time_checks + time_iterator) = MPI_Wtime(); // start

            if (process_rank == MASTER_PROCESS) {
                logtime();
                std::cout << "Reading training dataset starts at time " << *(time_checks + time_iterator) << std::endl;
            }
            ++time_iterator;

#endif

            Dataset df_train = read_dataset(filepath_training, rows_t, columns, target_column);

#if PERFORMANCE_CHECK

            MPI_Barrier(MPI_COMM_WORLD);
            *(time_checks + time_iterator) = MPI_Wtime(); // start

            if (process_rank == MASTER_PROCESS) {

                logtime();
                std::cout << "It took " << *(time_checks + time_iterator) - *(time_checks + time_iterator - 1)
                          << " seconds\n" << std::endl;

                logtime();
                std::cout << "svm preparation starts at time " << *(time_checks + time_iterator) << std::endl;
            }
            ++time_iterator;

#endif

            // TODO: change to parallel logic

            Kernel_SVM svm;

            svm.verbose = verbose;
            set_kernel_function(&svm, ker_type);

            double params[4] = {Cost, gamma, coef0, degree};


#if PERFORMANCE_CHECK

            MPI_Barrier(MPI_COMM_WORLD);
            *(time_checks + time_iterator) = MPI_Wtime(); // start

            if (process_rank == MASTER_PROCESS) {

                logtime();
                std::cout << "It took " << *(time_checks + time_iterator) - *(time_checks + time_iterator - 1)
                          << " seconds\n" << std::endl;
                logtime();
                std::cout << "Training starts at time " << *(time_checks + time_iterator) << std::endl;
            }
            ++time_iterator;

#endif

            train(df_train, &svm, params, lr, limit, true, save_dir_path, 0, eps);

#if PERFORMANCE_CHECK

            MPI_Barrier(MPI_COMM_WORLD);
            *(time_checks + time_iterator) = MPI_Wtime(); // start

            if (process_rank == MASTER_PROCESS) {

                logtime();
                std::cout << "It took " << *(time_checks + time_iterator) - *(time_checks + time_iterator - 1)
                          << " seconds\n" << std::endl;

                // std::cout << "Training starts at time " <<  *(time_checks + time_iterator) << std::endl;
            }
            ++time_iterator;

#endif

            break;
        }

            /// Testing case

        case train_flag::testing: {

/* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- */

            Dataset df_test = read_dataset(filepath_validation, 30, 5, 5);

            if (process_rank == MASTER_PROCESS) {

                Kernel_SVM svm;

                std::string saved_model_path = save_dir_path + filepath_svm;

                read_svm(&svm, saved_model_path);

                test(df_test, &svm);

                logtime();
                std::cout << "Testing completed." << std::endl;

                break;

            }

        }

            /// Tuning case

        case train_flag::tuning: {

/* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- */

            if (process_rank == MASTER_PROCESS) {
                logtime();
                std::cout << "Training Dataset filepath: " << filepath_training << std::endl;
                logtime();
                std::cout << "The dataset has " << rows_t << " rows and " << columns << " columns." << std::endl;
                logtime();
                std::cout << "Validation Dataset filepath: " << filepath_validation << std::endl;
                logtime();
                std::cout << "The dataset has " << rows_v << " rows and " << columns << " columns." << std::endl;
            }


#if CLI_ARGS

            double* cost_array;
            double* gamma_array;
            double* coef0_array;
            double* degree_array;

            int cost_array_size = 0, gamma_array_size = 0, coef0_array_size = 0, degree_array_size = 0;

            read_hyperparameters(filepath_hyperparameters, cost_array, &cost_array_size, gamma_array, &gamma_array_size, coef0_array, &coef0_array_size, degree_array, &degree_array_size);

#else

            size_t cost_array_size = 10, gamma_array_size = 8, coef0_array_size = 6, degree_array_size = 7;
            double cost_array[] = {0.001, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 100};
            double gamma_array[] = {0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10};
            double coef0_array[] = {0, 0.5, 1, 2.5, 5, 10};
            double degree_array[] = {1, 2, 3, 4, 5, 10, static_cast<double>(columns - 1)};

#endif
#if PERFORMANCE_CHECK

            MPI_Barrier(MPI_COMM_WORLD);
            *(time_checks + time_iterator) = MPI_Wtime();

            if (process_rank == MASTER_PROCESS) {
                logtime();
                std::cout << "\nAllocating inital vectors took "
                          << *(time_checks + time_iterator) - *(time_checks + time_iterator - 1) << " seconds\n"
                          << std::endl;
            }

            if (process_rank == MASTER_PROCESS) {
                logtime();
                std::cout << "Training dataset read starts at time " << *(time_checks + time_iterator) << "\n"
                          << std::endl;
            }
            ++time_iterator; // start tr dataset read
#endif

            Dataset df_train = read_dataset(filepath_training, rows_t, columns, target_column);

#if PERFORMANCE_CHECK

            MPI_Barrier(MPI_COMM_WORLD);
            *(time_checks + time_iterator) = MPI_Wtime(); // end of allocations

            if (process_rank == MASTER_PROCESS) {
                logtime();
                std::cout << "\nReading training dataset took a total of "
                          << *(time_checks + time_iterator) - *(time_checks + time_iterator - 1) << " seconds\n"
                          << std::endl;
            }

            if (process_rank == MASTER_PROCESS) {
                logtime();
                std::cout << "Validation dataset read starts at time " << *(time_checks + time_iterator) << "\n"
                          << std::endl;
            }
            ++time_iterator; // start val dataset read

#endif

            Dataset df_validation = read_dataset(filepath_validation, rows_v, columns, target_column);

#if PERFORMANCE_CHECK

            MPI_Barrier(MPI_COMM_WORLD);
            *(time_checks + time_iterator) = MPI_Wtime(); // end of allocations

            if (process_rank == MASTER_PROCESS) {
                logtime();
                std::cout << "\nReading validation dataset took "
                          << *(time_checks + time_iterator) - *(time_checks + time_iterator - 1) << " seconds\n"
                          << std::endl;
            }
            if (process_rank == MASTER_PROCESS) {
                logtime();
                std::cout << "Memory allocation for tuning starts at time " << *(time_checks + time_iterator) << "\n"
                          << std::endl;
            }
            ++time_iterator;

#endif

            // TODO: refactor?

            size_t linear_rows = cost_array_size;
            size_t radial_rows = cost_array_size * gamma_array_size;
            size_t sigmoid_rows = cost_array_size * gamma_array_size * coef0_array_size;
            size_t polynomial_rows = cost_array_size * gamma_array_size * coef0_array_size * degree_array_size;

            size_t tuning_table_rows = linear_rows + radial_rows + sigmoid_rows + polynomial_rows;
            size_t tuning_table_columns = NUMBER_OF_HYPER_PARAMETERS + 1/* accuracy*/ + 1/*class 1 accuracy*/ +
                                          1/*class 2 accuracy*/; // NB: type of kernel will be printed separately
            if (process_rank == MASTER_PROCESS) {
                std::cout << "There are a total of " << IMPLEMENTED_KERNELS << " kernels to tune." << std::endl;
                std::cout << "Tuning will use:\n\t " << cost_array_size << " different costs, " << std::endl;
                std::cout << "\t " << gamma_array_size << " different gamma, " << std::endl;
                std::cout << "\t " << coef0_array_size << " different intercepts, " << std::endl;
                std::cout << "\t " << degree_array_size << " different exponential degrees, " << std::endl;
                std::cout << "For a total of " << tuning_table_rows << " combinations" << std::endl;
            }


            auto *final_tuning_table = (double *) calloc(tuning_table_rows * tuning_table_columns,
                                                         sizeof(double)); // matrix


#if DEBUG_MPI_OPERATIONS

            print_matrix(final_tuning_table, tuning_table_rows, tuning_table_columns, true);
    std::cout << "\n\n\n\n\n\n\n\n\n\n" ;

#endif

            auto *local_tuning_table = (double *) calloc(tuning_table_rows * tuning_table_columns,
                                                         sizeof(double)); // matrix


#if DEBUG_MPI_OPERATIONS


            print_matrix(local_tuning_table, tuning_table_rows, tuning_table_columns, true);

#endif


            char kernel_type_final_table[tuning_table_rows];
            memset(kernel_type_final_table, 'l', linear_rows);
            memset(kernel_type_final_table + linear_rows, 'r', radial_rows);
            memset(kernel_type_final_table + linear_rows + radial_rows, 's', sigmoid_rows);
            memset(kernel_type_final_table + linear_rows + radial_rows + sigmoid_rows, 'p', polynomial_rows);


            // TODO: decide condition
            if (world_size < 100) {
                // one after the other

#if PERFORMANCE_CHECK

                MPI_Barrier(MPI_COMM_WORLD);
                *(time_checks + time_iterator) = MPI_Wtime(); // end of allocations

                if (process_rank == MASTER_PROCESS) {
                    std::cout << "\n Latest memory allocation ends at time " << *(time_checks + time_iterator)
                              << std::endl;
                    std::cout << "It took " << *(time_checks + time_iterator) - *(time_checks + time_iterator - 1)
                              << " seconds\n" << std::endl;

                }
                ++time_iterator; // start linear tuning

#endif

                //linear
                if (process_rank == MASTER_PROCESS) {
                    std::cout << "Starting linear tuning" << std::endl;
                }
                tune_linear(&df_train, &df_validation, cost_array, cost_array_size, local_tuning_table, 0,
                            tuning_table_columns, MASTER_PROCESS, world_size);
                std::cout << "Process " << process_rank << " has finished linear tuning" << std::endl;
                //radial

#if PERFORMANCE_CHECK

                MPI_Barrier(MPI_COMM_WORLD);
                *(time_checks + time_iterator) = MPI_Wtime();

                if (process_rank == MASTER_PROCESS) {
                    std::cout << "\n linear tuning ends at time " << *(time_checks + time_iterator) << std::endl;
                    std::cout << "\n It took " << *(time_checks + time_iterator) - *(time_checks + time_iterator - 1)
                              << "seconds\n" << std::endl;
                }
                ++time_iterator; // start radial tuning

#endif

                if (process_rank == MASTER_PROCESS) {
                    std::cout << "Starting radial tuning" << std::endl;
                }
                //tune_radial(&df_train, &df_validation, cost_array, cost_array_size, gamma_array, gamma_array_size,local_tuning_table, linear_rows * tuning_table_columns, tuning_table_columns, MASTER_PROCESS, world_size);
                std::cout << "Process " << process_rank << " has finished radial tuning" << std::endl;
                //sigmoid

#if PERFORMANCE_CHECK

                MPI_Barrier(MPI_COMM_WORLD);
                *(time_checks + time_iterator) = MPI_Wtime();

                if (process_rank == MASTER_PROCESS) {
                    std::cout << "\n radial tuning ends at time " << *(time_checks + time_iterator) << std::endl;
                    std::cout << "\n It took " << *(time_checks + time_iterator) - *(time_checks + time_iterator - 1)
                              << "seconds\n" << std::endl;
                }
                ++time_iterator; // start sigmoid tuning

#endif

                if (process_rank == MASTER_PROCESS) {
                    std::cout << "Starting sigmoid tuning" << std::endl;
                }
                // tune_sigmoid(&df_train, &df_validation, cost_array, cost_array_size, gamma_array, gamma_array_size, coef0_array, coef0_array_size, local_tuning_table, linear_rows * tuning_table_columns + radial_rows * tuning_table_columns, tuning_table_columns, MASTER_PROCESS, world_size);
                std::cout << "Process " << process_rank << " has finished sigmoid tuning" << std::endl;
                //polynomial

#if PERFORMANCE_CHECK

                MPI_Barrier(MPI_COMM_WORLD);
                *(time_checks + time_iterator) = MPI_Wtime();

                if (process_rank == MASTER_PROCESS) {
                    std::cout << "\n sigmoid tuning ends at time " << *(time_checks + time_iterator) << std::endl;
                    std::cout << "\n It took " << *(time_checks + time_iterator) - *(time_checks + time_iterator - 1)
                              << "seconds\n" << std::endl;
                }
                ++time_iterator; // start polynomial tuning

#endif

                if (process_rank == MASTER_PROCESS) {
                    std::cout << "Starting polynomial tuning" << std::endl;
                }
                // tune_polynomial(&df_train, &df_validation, cost_array, cost_array_size, gamma_array, gamma_array_size, coef0_array, coef0_array_size,degree_array, degree_array_size, local_tuning_table, linear_rows * tuning_table_columns + radial_rows * tuning_table_columns +sigmoid_rows * tuning_table_columns , tuning_table_columns, MASTER_PROCESS, world_size);
                std::cout << "Process " << process_rank << " has finished polynomial tuning" << std::endl;

#if PERFORMANCE_CHECK

                MPI_Barrier
                        (MPI_COMM_WORLD);
                *(time_checks + time_iterator) = MPI_Wtime();

                if (process_rank == MASTER_PROCESS) {
                    std::cout << "\n polynomial tuning ends at time " << *(time_checks + time_iterator) << std::endl;
                    std::cout << "\n It took " << *(time_checks + time_iterator) - *(time_checks + time_iterator - 1)
                              << "seconds\n" << std::endl;
                    std::cout << "\nStarting reduce\n" << std::endl;
                }
                ++time_iterator; // start reduce

#endif

                MPI_Reduce(local_tuning_table, final_tuning_table, (int) (tuning_table_rows * tuning_table_columns),
                           MPI_DOUBLE, MPI_SUM, MASTER_PROCESS, MPI_COMM_WORLD);
                free(local_tuning_table);

#if PERFORMANCE_CHECK

                MPI_Barrier(MPI_COMM_WORLD);
                *(time_checks + time_iterator) = MPI_Wtime();

                if (process_rank == MASTER_PROCESS) {
                    std::cout << "\n Reduce ends at time " << *(time_checks + time_iterator) << std::endl;
                    std::cout << "\n It took " << *(time_checks + time_iterator) - *(time_checks + time_iterator - 1)
                              << "seconds\n" << std::endl;
                }
                ++time_iterator; // start calculating accuracy

#endif
#if DEBUG_MPI_OPERATIONS

                print_matrix(final_tuning_table, tuning_table_rows, tuning_table_columns);

#endif

            } else {
                // together, will require a gather
            }
            if (process_rank == MASTER_PROCESS) {
                double accuracies[tuning_table_rows];
                get_column(final_tuning_table, NUMBER_OF_HYPER_PARAMETERS, tuning_table_columns, tuning_table_rows,
                           accuracies);
                auto *m = std::max_element(accuracies, accuracies + tuning_table_rows);
                int row_index = (int) (m - accuracies);
                double max_accuracy = *m;
                std::cout << "Best Accuracy: " << max_accuracy << std::endl;
                double best_row[tuning_table_columns];
                get_row(final_tuning_table, row_index, tuning_table_columns, best_row);


                std::cout << "Best combination:\n\tKernel Cost Gamma Coef0 Degree Accuracy C1_Acc. C2_Acc."
                          << std::endl;
                switch (kernel_type_final_table[row_index]) {
                    case 'l': {
                        std::cout << "\tLinear ";
                        break;
                    }
                    case 'r': {
                        std::cout << "\tRadial ";
                        break;
                    }
                    case 's': {
                        std::cout << "\tSigmoid ";
                        break;
                    }
                    case 'p': {
                        std::cout << "\tPolynomial ";
                        break;
                    }

                }
                print_vector(best_row, tuning_table_columns, false);
                // todo: save data somewhere
            }

            free(final_tuning_table);

        }
            // end of tuning part
    }
    // end of switch case


#if PERFORMANCE_CHECK

    MPI_Barrier(MPI_COMM_WORLD);
    *(time_checks + time_iterator) = MPI_Wtime();

    if (process_rank == MASTER_PROCESS) {
        logtime();
        std::cout << "Program ends at time " << *(time_checks + time_iterator) << std::endl;
        logtime();
        std::cout << "Last step took " << *(time_checks + time_iterator) - *(time_checks + time_iterator - 1)
                  << "seconds\n" << std::endl;
    }
    // ++time_iterator; // end

    free(time_checks);

#endif

    MPI_Finalize();

    return 0;

}