#include <iostream>
#include <mpi.h>
#include <getopt.h>
#include <sys/time.h>
#include <math.h>

#include "Dataset.h"
#include "tune_svm.h"
#include "read_dataset.h"


#define CLI_ARGS true
#define IMPLEMENTED_KERNELS 4
#define NUMBER_OF_HYPER_PARAMETERS 4
#define NUMBER_OF_PERFORMANCE_CHECKS 15
#define SHOW_LOGTIME true
#define DEBUG_MAIN false
#define MAX_HP_VALUES 10

#if SHOW_LOGTIME

void logtime() {

    int process_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);

    char buffer[26];
    long millisec;
    struct tm *tm_info;
    struct timeval tv;

    gettimeofday(&tv, nullptr);


    millisec = lrint((double) tv.tv_usec / 1000.0); // Round to nearest millisec
    if (millisec >= 1000) { // Allow for rounding up to nearest second

        millisec -= 1000;
        tv.tv_sec++;
    }

    tm_info = localtime(&tv.tv_sec);

    strftime(buffer, 26, "%Y-%m-%d %H:%M:%S", tm_info);


    printf("[rank %d at %s.%03ld] ", process_rank, buffer, millisec);

}

#endif

enum train_flag {
    training = 0, testing = 1, tuning = 2
} flag;


//  TODO:
//        implement:
//                  open_mp
//                  mpi logic for train and test
//        debug:
//                  debug cli args
//                  svm test


#if CLI_ARGS
const char *program_name;

void print_usage(FILE *stream, int exit_code) {
    std::cout << "Usage:  " << program_name << " options [ inputfile ... ]\n"
              << "  -h  --help               Display this usage information.\n"
              << "  -l  --logic              Program logic, may be training, testing or tuning.\n"
              << "  -p  --parallel-tuning    Set tuning logic to parallel (default is split).\n"
              << "  -i  --path1              First input path supplied, may be interpreted as training path or testing path.\n"
              << "  -I  --path2              Second input path supplied, in tuning logic is interpreted as validation.\n"
              << "  -t  --target_column      Index of the target column.\n"
              << "  -c  --columns            Number of columns in the dataset.\n"
              << "  -r  --row1               Number of rows in the first supplied dataset.\n"
              << "  -R  --row2               Number of rows in the second supplied dataset.\n"
              << "  -H  --hparameters_path   Path to the hyperparameters file.\n"
              << "  -s  --svm_path           Path to the SVM file.\n"
              << "  -k  --kernel             Kernel type, may be l (linear), polynomial (p), rbf (r) or sigmoid (s).\n"
              << "  -C  --cost               Cost parameter.\n"
              << "  -g  --gamma              Gamma parameter.\n"
              << "  -O  --coef0              Coef0 parameter.\n"
              << "  -d  --degree             Degree parameter.\n"
              << "  -T  --learning_rate      Learning rate parameter.\n"
              << "  -E  --eps                Epsilon parameter.\n"
              << "  -L  --limit              Limit parameter.\n"
              << "  -v  --verbose            Print verbose messages.\n"
              << std::endl;
    exit(exit_code);
}

#endif

int main(int argc, char *argv[]) {

    /* Initialize MPI ------------------------------------ */

    MPI_Init(nullptr, nullptr);  // TODO : check argc & argv

    // Get the rank of the process
    int process_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    /* --------------------------------------------------- */

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


#if CLI_ARGS

    int next_option;

    /* A string listing valid short options letters. */
    const char *const short_options = "hl:i:I:t:c:r:R:H:s:k:C:g:O:d:T:E:L:v";

    /* An array describing valid long options.  */
    const struct option long_options[] = {
            {"help",             0, NULL, 'h'},
            {"logic",            0, NULL, 'l'},
            {"parallel-tuning",  0, NULL, 'p'},
            {"path1",            0, NULL, 'i'},
            {"path2",            0, NULL, 'I'},
            {"target_column",    0, NULL, 't'},
            {"columns",          0, NULL, 'c'},
            {"row1",             0, NULL, 'r'},
            {"row2",             0, NULL, 'R'},
            {"hparameters_path", 0, NULL, 'H'},
            {"svm_path",         0, NULL, 's'},
            {"kernel",           0, NULL, 'k'},
            {"cost",             0, NULL, 'C'},
            {"gamma",            0, NULL, 'g'},
            {"coef0",            0, NULL, 'O'},
            {"degree",           0, NULL, 'd'},
            {"learning_rate",    0, NULL, 'T'},
            {"eps",              0, NULL, 'E'},
            {"limit",            0, NULL, 'L'},
            {"verbose",          0, NULL, 'v'},
            {NULL,               0, NULL, 0}
    };

    /**
     * Parameters initialization
     */

    bool tuning_logic = false;

    std::string filepath_training;
    std::string filepath_validation;
    std::string filepath_testing;
    std::string filepath_svm;

    int rows_t = 0;
    int rows_v = 0;
    int target_column = 0;
    int columns = 0;

    std::string hparameters_path;

    std::string save_dir_path = "/home/azel/Developer/hpc2022/saved_svm";  // TODO : what to do with this???

    char ker_type = '\0';
    double Cost = 0.0;
    double gamma = 0.0;
    double coef0 = 0.0;
    double degree = 0.0;
    double lr = 0.0;
    double limit = 0.01;
    double eps = DEFAULT_EPS;

    int verbose = 0;

    program_name = argv[0];


    /**
     * CLI arguments parsing
     */

    {

        do {

            next_option = getopt_long(argc, argv, short_options, long_options, NULL);

            switch (next_option) {

                case 'h':   /* -h or --help */
                    print_usage(stdout, 0);

                case 'l':   /* -l or --logic */
                    if (strcmp(optarg, "training") == 0) {
                        // flag == training;  // unnecessary
#if SHOW_LOGTIME
                        logtime();
#endif
                        std::cout << "[INFO] Logic set to training." << std::endl;
                    } else if (strcmp(optarg, "testing") == 0) {
                        flag = testing;
#if SHOW_LOGTIME
                        logtime();
#endif
                        std::cout << "[INFO] Logic set to training." << std::endl;
                    } else if (strcmp(optarg, "tuning") == 0) {
                        flag = tuning;
#if SHOW_LOGTIME
                        logtime();
#endif
                        std::cout << "[INFO] Logic set to training." << std::endl;
                    } else {
#if SHOW_LOGTIME
                        logtime();
#endif
                        std::cout << "[ERROR] Invalid logic argument, please use training, testing or tuning.\n";
                        print_usage(stdout, 0);
                    }
                    break;

                case 'p':   /* -p or --parallel-tuning */
                    tuning_logic = true;
                    if (tuning_logic == true) {
#if SHOW_LOGTIME
                        logtime();
#endif
                        std::cout
                                << "[INFO] Tuning logic set to parallel. All processes will go through parallel training.\n";
                    } else {
#if SHOW_LOGTIME
                        logtime();
#endif
                        std::cout
                                << "[INFO] Tuning logic set to split by default. Processes will split the available combination.\n";
                    }
                    break;

                case 'i':   /* -i or --path1 */
                    if ((flag == training) || (flag == tuning)) {
                        filepath_training = optarg;
#if SHOW_LOGTIME
                        logtime();
#endif
                        std::cout << "[INFO] Training file path set to " << filepath_training << std::endl;
                    } else if (flag == testing) {
                        filepath_validation = optarg;
#if SHOW_LOGTIME
                        logtime();
#endif
                        std::cout << "[INFO] Testing file path set to " << filepath_validation << std::endl;
                    } else {
#if SHOW_LOGTIME
                        logtime();
#endif
                        std::cout << "[ERROR] Something went wrong setting dataset path.\n";
                        print_usage(stdout, 0);
                    }
                    break;

                case 'I':   /* -I or --path2 */
                    if (flag == tuning) {
                        filepath_validation = optarg;
#if SHOW_LOGTIME
                        logtime();
#endif
                        std::cout << "[INFO] Validation file path set to " << filepath_validation << std::endl;
                    } else {
#if SHOW_LOGTIME
                        logtime();
#endif
                        std::cout << "[ERROR] Training and testing logics only require 1 dataset path argument.\n";
                        print_usage(stdout, 0);
                    }
                    break;

                case 't':   /* -t or --target_column */
                    target_column = std::atoi(optarg);
#if SHOW_LOGTIME
                    logtime();
#endif
                    std::cout << "[INFO] Target column set to " << target_column << std::endl;
                    break;

                case 'c':   /* -c or --columns */
                    columns = std::atoi(optarg);
#if SHOW_LOGTIME
                    logtime();
#endif
                    std::cout << "[INFO] Number of columns set to " << columns << std::endl;
                    break;

                case 'r':   /* -r or --row1 */

                    if ((flag == training) || (flag == tuning)) {
                        rows_t = std::atoi(optarg);
#if SHOW_LOGTIME
                        logtime();
#endif
                        std::cout << "[INFO] Number of rows for training dataset set to " << rows_t << std::endl;
                    } else if (flag == testing) {
                        rows_v = std::atoi(optarg);
#if SHOW_LOGTIME
                        logtime();
#endif
                        std::cout << "[INFO] Number of rows for testing dataset set to " << rows_v << std::endl;
                    } else {
#if SHOW_LOGTIME
                        logtime();
#endif
                        std::cout << "[ERROR] Something went wrong setting rows number.\n";
                        print_usage(stdout, 0);
                    }

                    break;

                case 'R':   /* -R or --row2 */

                    if (flag == tuning) {
                        rows_v = std::atoi(optarg);
#if SHOW_LOGTIME
                        logtime();
#endif
                        std::cout << "[INFO] Number of rows for validation dataset set to " << rows_t << std::endl;
                    } else {
#if SHOW_LOGTIME
                        logtime();
#endif
                        std::cout << "[ERROR] Training and testing logics only require 1 rows number argument.\n";
                        print_usage(stdout, 0);
                    }

                    break;

                case 'H':   /* -H or --hparameters_path */
                    hparameters_path = optarg;
                    break;

                case 's':   /* -s or --svm_path */
                    filepath_svm = optarg;
#if SHOW_LOGTIME
                    logtime();
#endif
                    std::cout << "[INFO] SVM file path set to " << filepath_svm << std::endl;
                    break;

                case 'k':   /* -k or --kernel */
                    ker_type = *optarg;
#if SHOW_LOGTIME
                    logtime();
#endif
                    std::cout << "[INFO] Kernel type set to `" << ker_type << "`" << std::endl;
                    break;

                case 'C':   /* -C or --cost */
                    Cost = std::stoi(optarg);
#if SHOW_LOGTIME
                    logtime();
#endif
                    std::cout << "[INFO] Cost parameter set to " << Cost << std::endl;
                    break;

                case 'g':   /* -g or --gamma */
                    gamma = std::stoi(optarg);
#if SHOW_LOGTIME
                    logtime();
#endif
                    std::cout << "[INFO] Gamma parameter set to " << gamma << std::endl;
                    break;

                case 'O':   /* -O or --coef0 */
                    coef0 = std::stoi(optarg);
#if SHOW_LOGTIME
                    logtime();
#endif
                    std::cout << "[INFO] Coef0 parameter set to " << coef0 << std::endl;
                    break;

                case 'd':   /* -d or --degree */
                    degree = std::stoi(optarg);
#if SHOW_LOGTIME
                    logtime();
#endif
                    std::cout << "[INFO] Degree parameter set to " << degree << std::endl;
                    break;

                case 'T':   /* -T or --learning_rate */
                    lr = std::stoi(optarg);
#if SHOW_LOGTIME
                    logtime();
#endif
                    std::cout << "[INFO] Learning rate set to " << lr << std::endl;
                    break;

                case 'E':   /* -E or --eps */
                    eps = std::stoi(optarg);
#if SHOW_LOGTIME
                    logtime();
#endif
                    std::cout << "[INFO] Epsilon set to " << eps << std::endl;
                    break;

                case 'L':   /* -L or --limit */
                    limit = std::stoi(optarg);
#if SHOW_LOGTIME
                    logtime();
#endif
                    std::cout << "[INFO] Limit value set to " << limit << std::endl;
                    break;

                case 'v':   /* -v or --verbose */
                    verbose = 1;
                    break;

                case '?':   /* The user specified an invalid option */
#if SHOW_LOGTIME
                    logtime();
#endif
                    std::cout << "\n[WARN] You entered an invalid option." << std::endl;
                    print_usage(stderr, 1);

                case -1:    /* Done with options */
                    break;

                default:    /* Something else unexpected */
                    abort();

            }

        } while (next_option != -1);

#if SHOW_LOGTIME
        logtime();
#endif
        std::cout << "[INFO] Cli arguments successfully parsed.\n" << std::endl;

    }

#else

    /**
     * Flag selection (training, testing, tuning)
     */


    flag = tuning;

    bool tuning_logic = false;

    /* Antonio */
     std::string filepath_training = "/Users/azel/Developer/hpc2022/data/iris_train.csv";
     std::string filepath_validation = "/Users/azel/Developer/hpc2022/data/iris_validation.csv";
     std::string save_dir_path = "/Users/azel/Developer/hpc2022/saved_svm/";


    /* Maurizio */
//    std::string filepath_training = "/home/dmmp/Documents/GitHub/hpc2022/data/iris_train.csv";
//    std::string filepath_validation = "/home/dmmp/Documents/GitHub/hpc2022/data/iris_validation.csv";
//    std::string save_dir_path = "/home/dmmp/Documents/GitHub/hpc2022/saved_svm";


    std::string filepath_hyper_parameters = "../data/hyperparameters.csv"; // TODO: implement
    std::string filepath_svm = "/Users/azel/Developer/hpc2022/saved_svm/sigmoids_C0.500000_G0.010000_O0.000000.svm";

    int rows_t = 70, rows_v = 30, columns = 5, target_column = 5;
    char ker_type = 's';

    bool verbose = false;

    double Cost = 0.5;
    double gamma = 0.01;
    double coef0 = 0;
    double degree = 0;

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
#if SHOW_LOGTIME
        logtime();
#endif
        std::cout << "Program starts at time " << *(time_checks + time_iterator) << "\n" << std::endl;
    }
    ++time_iterator;

#endif

    /**
     * Switch block begins
     */


    switch (flag) {


        /// Training case

        case train_flag::training : {

// -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#if PERFORMANCE_CHECK

            MPI_Barrier(MPI_COMM_WORLD);
            *(time_checks + time_iterator) = MPI_Wtime(); // start

            if (process_rank == MASTER_PROCESS) {
#if SHOW_LOGTIME
                logtime();
#endif
                std::cout << "Reading training dataset starts at time " << *(time_checks + time_iterator) << std::endl;
            }
            ++time_iterator;

#endif

            Dataset df_train = read_dataset(filepath_training, rows_t, columns, target_column);

#if PERFORMANCE_CHECK

            MPI_Barrier(MPI_COMM_WORLD);
            *(time_checks + time_iterator) = MPI_Wtime(); // start

            if (process_rank == MASTER_PROCESS) {

#if SHOW_LOGTIME
                logtime();
#endif
                std::cout << "It took " << *(time_checks + time_iterator) - *(time_checks + time_iterator - 1)
                          << " seconds\n" << std::endl;

#if SHOW_LOGTIME
                logtime();
#endif
                std::cout << "svm preparation starts at time " << *(time_checks + time_iterator) << std::endl;
            }
            ++time_iterator;

#endif


            Kernel_SVM svm;

            svm.verbose = verbose;
            set_kernel_function(&svm, ker_type);

            double params[4] = {Cost, gamma, coef0, degree};


#if PERFORMANCE_CHECK

            MPI_Barrier(MPI_COMM_WORLD);
            *(time_checks + time_iterator) = MPI_Wtime(); // start

            if (process_rank == MASTER_PROCESS) {

#if SHOW_LOGTIME
                logtime();
#endif
                std::cout << "It took " << *(time_checks + time_iterator) - *(time_checks + time_iterator - 1)
                          << " seconds\n" << std::endl;
#if SHOW_LOGTIME
                logtime();
#endif
                std::cout << "Training starts at time " << *(time_checks + time_iterator) << std::endl;
            }
            ++time_iterator;

#endif


            parallel_train(df_train, &svm, params, lr, limit, MASTER_PROCESS, world_size, true, save_dir_path, 0, eps);

#if PERFORMANCE_CHECK

            MPI_Barrier(MPI_COMM_WORLD);
            *(time_checks + time_iterator) = MPI_Wtime(); // start

            if (process_rank == MASTER_PROCESS) {

#if SHOW_LOGTIME
                logtime();
#endif
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

            Kernel_SVM svm;

            read_svm(&svm, filepath_svm);

            svm.verbose = true;

            parallel_test(df_test, &svm, MASTER_PROCESS, world_size);

            break;

        }

            /// Tuning case

        case train_flag::tuning: {

/* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- */

            if (process_rank == MASTER_PROCESS) {
#if SHOW_LOGTIME
                logtime();
#endif
                std::cout << "Training Dataset filepath: " << filepath_training << std::endl;

                std::cout << "The dataset has " << rows_t << " rows and " << columns << " columns." << std::endl;

                std::cout << "Validation Dataset filepath: " << filepath_validation << std::endl;

                std::cout << "The dataset has " << rows_v << " rows and " << columns << " columns." << std::endl;
            }


#if CLI_ARGS

            double *cost_array;
            double *gamma_array;
            double *coef0_array;
            double *degree_array;

            cost_array = (double *) malloc(sizeof(double) * MAX_HP_VALUES);
            gamma_array = (double *) malloc(sizeof(double) * MAX_HP_VALUES);
            coef0_array = (double *) malloc(sizeof(double) * MAX_HP_VALUES);
            degree_array = (double *) malloc(sizeof(double) * MAX_HP_VALUES);

            int cost_array_size = 0, gamma_array_size = 0, coef0_array_size = 0, degree_array_size = 0;

            read_hyperparameters(hparameters_path,
                                 cost_array, cost_array_size,
                                 gamma_array, gamma_array_size,
                                 coef0_array, coef0_array_size,
                                 degree_array, degree_array_size);


            if (process_rank == MASTER_PROCESS) {

                std::cout << "\n\nCost array size: " << cost_array_size << std::endl;
                std::cout << "Cost array: " << std::endl;
                for (int i = 0; i < cost_array_size; ++i) {
                    std::cout << cost_array[i] << " ";
                }

                std::cout << "\n\nGamma array size: " << gamma_array_size << std::endl;
                std::cout << "Gamma array: " << std::endl;
                for (int i = 0; i < gamma_array_size; ++i) {
                    std::cout << gamma_array[i] << " ";
                }

                std::cout << "\n\nCoef0 array size: " << coef0_array_size << std::endl;
                std::cout << "Coef0 array: " << std::endl;
                for (int i = 0; i < coef0_array_size; ++i) {
                    std::cout << coef0_array[i] << " ";
                }

                std::cout << "\n\nDegree array size: " << degree_array_size << std::endl;
                std::cout << "Degree array: " << std::endl;
                for (int i = 0; i < degree_array_size; ++i) {
                    std::cout << degree_array[i] << " ";
                }

            }

#else

            int cost_array_size = 6, gamma_array_size = 8, coef0_array_size = 6, degree_array_size = 7;
            double cost_array[] = {0.001, 0.01, 0.05, 0.1, 0.5, 1};//, 2, 5, 10, 100};
            double gamma_array[] = {0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10};
            double coef0_array[] = {0, 0.5, 1, 2.5, 5, 10};
            double degree_array[] = {1, 2, 3, 4, 5, 10, static_cast<double>(columns - 1)};

#endif
#if PERFORMANCE_CHECK

            MPI_Barrier(MPI_COMM_WORLD);
            *(time_checks + time_iterator) = MPI_Wtime();

            if (process_rank == MASTER_PROCESS) {
#if SHOW_LOGTIME
                logtime();
#endif
                std::cout << "\nAllocating inital vectors took "
                          << *(time_checks + time_iterator) - *(time_checks + time_iterator - 1) << " seconds\n"
                          << std::endl;
            }

            if (process_rank == MASTER_PROCESS) {
#if SHOW_LOGTIME
                logtime();
#endif
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
#if SHOW_LOGTIME
                logtime();
#endif
                std::cout << "\nReading training dataset took a total of "
                          << *(time_checks + time_iterator) - *(time_checks + time_iterator - 1) << " seconds\n"
                          << std::endl;
            }

            if (process_rank == MASTER_PROCESS) {
#if SHOW_LOGTIME
                logtime();
#endif
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
#if SHOW_LOGTIME
                logtime();
#endif
                std::cout << "\nReading validation dataset took "
                          << *(time_checks + time_iterator) - *(time_checks + time_iterator - 1) << " seconds\n"
                          << std::endl;
            }
            if (process_rank == MASTER_PROCESS) {
#if SHOW_LOGTIME
                logtime();
#endif
                std::cout << "Memory allocation for tuning starts at time " << *(time_checks + time_iterator) << "\n"
                          << std::endl;
            }
            ++time_iterator;

#endif

            // TODO: refactor?

            int linear_rows = cost_array_size;
            int radial_rows = cost_array_size * gamma_array_size;
            int sigmoid_rows = cost_array_size * gamma_array_size * coef0_array_size;
            int polynomial_rows = cost_array_size * gamma_array_size * coef0_array_size * degree_array_size;

            int tuning_table_rows = linear_rows + radial_rows + sigmoid_rows + polynomial_rows;
            int tuning_table_columns = NUMBER_OF_HYPER_PARAMETERS + 1/* accuracy*/ + 1/*class 1 accuracy*/ +
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

            auto *local_tuning_table = (double *) calloc(tuning_table_rows * tuning_table_columns,
                                                         sizeof(double)); // matrix




            if (tuning_logic) {
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
                //tune_linear(&df_train, &df_validation, cost_array, cost_array_size, local_tuning_table, 0,tuning_table_columns, MASTER_PROCESS, world_size,lr,limit,eps,verbose);
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
                //tune_radial(&df_train, &df_validation, cost_array, cost_array_size, gamma_array, gamma_array_size,local_tuning_table, linear_rows, tuning_table_columns, MASTER_PROCESS, world_size,lr,limit,eps,verbose);
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
                tune_sigmoid(&df_train, &df_validation, cost_array, cost_array_size, gamma_array, gamma_array_size,
                             coef0_array, coef0_array_size, local_tuning_table, linear_rows + radial_rows,
                             tuning_table_columns, MASTER_PROCESS, world_size);
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
                //tune_polynomial(&df_train, &df_validation, cost_array, cost_array_size, gamma_array, gamma_array_size, coef0_array, coef0_array_size,degree_array, degree_array_size, local_tuning_table, linear_rows + radial_rows +sigmoid_rows , tuning_table_columns, MASTER_PROCESS, world_size);
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
#if DEBUG_MAIN
                if(process_rank == MASTER_PROCESS){
                    print_matrix(final_tuning_table, tuning_table_rows, tuning_table_columns);
                }


#endif

            } else {
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
                //tune_linear2(&df_train, &df_validation, cost_array, cost_array_size, local_tuning_table, 0,tuning_table_columns, MASTER_PROCESS, world_size,lr,limit,eps,verbose);
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
                //tune_radial2(&df_train, &df_validation, cost_array, cost_array_size, gamma_array, gamma_array_size,local_tuning_table, linear_rows, tuning_table_columns, MASTER_PROCESS, world_size,lr,limit,eps,verbose);
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
                tune_sigmoid2(&df_train, &df_validation, cost_array, cost_array_size, gamma_array, gamma_array_size,
                              coef0_array, coef0_array_size, local_tuning_table, linear_rows + radial_rows,
                              tuning_table_columns, MASTER_PROCESS, world_size);
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
                //tune_polynomial2(&df_train, &df_validation, cost_array, cost_array_size, gamma_array, gamma_array_size, coef0_array, coef0_array_size,degree_array, degree_array_size, local_tuning_table, linear_rows + radial_rows +sigmoid_rows , tuning_table_columns, MASTER_PROCESS, world_size);
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

//                MPI_Reduce(local_tuning_table, final_tuning_table, (int) (tuning_table_rows * tuning_table_columns),
//                           MPI_DOUBLE, MPI_SUM, MASTER_PROCESS, MPI_COMM_WORLD);
//                free(local_tuning_table);

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
#if DEBUG_MAIN
                if(process_rank == MASTER_PROCESS){
                    print_matrix(final_tuning_table, tuning_table_rows, tuning_table_columns);
                }
#endif
            }
            if (process_rank == MASTER_PROCESS) {

                // todo: revise selection logic
                auto *accuracies = (double *) calloc(tuning_table_rows, sizeof(double));
                get_column(local_tuning_table, NUMBER_OF_HYPER_PARAMETERS, tuning_table_columns, tuning_table_rows,
                           accuracies);
                auto *m = std::max_element(accuracies, accuracies + tuning_table_rows);
                int row_index = (int) (m - accuracies);
                double max_accuracy = *m;
                std::cout << "Best Accuracy: " << max_accuracy << std::endl;
                auto *best_row = (double *) calloc(tuning_table_columns, sizeof(double));
                get_row(local_tuning_table, row_index, tuning_table_columns, best_row);


                std::cout << "Best combination:\n\tKernel Cost Gamma Coef0 Degree Accuracy C1_Acc. C2_Acc."
                          << std::endl;

                char out_kernel;

                if (row_index < linear_rows) {
                    out_kernel = 'l';
                } else if (row_index < linear_rows + radial_rows) {
                    out_kernel = 'r';
                } else if (row_index < linear_rows + radial_rows + sigmoid_rows) {
                    out_kernel = 's';
                } else if (row_index < linear_rows + radial_rows + sigmoid_rows + polynomial_rows) {
                    out_kernel = 'p';
                } else {
                    out_kernel = -1;
                }
                switch (out_kernel) {
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
                    default: {
                        std::cout << "\tError from the code, contact the developer!";
                        exit(2);
                    }

                }
                print_vector(best_row, tuning_table_columns, false);
                // todo: save data somewhere

                free(best_row);
                free(accuracies);
            }

            free(local_tuning_table);

        }

    }


#if PERFORMANCE_CHECK

    MPI_Barrier(MPI_COMM_WORLD);
    *(time_checks + time_iterator) = MPI_Wtime();

    if (process_rank == MASTER_PROCESS) {
#if SHOW_LOGTIME
        logtime();
#endif
        std::cout << "Program ends at time " << *(time_checks + time_iterator) << std::endl;
#if SHOW_LOGTIME
        logtime();
#endif
        std::cout << "Last step took " << *(time_checks + time_iterator) - *(time_checks + time_iterator - 1)
                  << " seconds\n" << std::endl;
    }
    // ++time_iterator; // end

    free(time_checks);

#endif

    MPI_Finalize();

    return 0;

}