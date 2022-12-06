#include <iostream>
#include <mpi.h>
#include <getopt.h>
#include <cmath>
#include "utils.h"
#include "Dataset.h"
#include "tune_svm.h"
#include "read_dataset.h"


#define CLI_ARGS true
#define IMPLEMENTED_KERNELS 4
#define NUMBER_OF_HYPER_PARAMETERS 4
#define NUMBER_OF_PERFORMANCE_CHECKS 20
#define SHOW_LOGTIME true
#define DEBUG_MAIN false
#define MAX_HP_VALUES 20 // HP are hyperparameters, not health points

std::string read_env_var(const std::string &name) {
    std::string ris;
    if (getenv(name.c_str())) {
        ris = getenv(name.c_str());
    } else {
        std::cout << "[WARNING] The environment variable" << name << "was not found.\n";
        // exit(1);
        return "FALSE";
    }

#if DEBUG_MAIN
    std::cout << name << ": " << ris << "\n\n";
#endif
    return ris;
}

// TODO: what about boost bint? wth?

enum train_flag {
    training = 0, testing = 1, tuning = 2, undefined = -1
} flag;


//  TODO:
//        next steps:
//                  hybrid open_mp
//        debug:
//                  cli args
//                  svm test
//                  mpi logic for train and test


#if CLI_ARGS


void print_usage(const std::string &program_name) {
    std::cout << "Usage:  " << program_name << " options [ inputfile ... ]\n"
              << "  -h  --help                   Display this usage information.\n"
              << "  -l  --logic                  Program logic, may be 'training', 'testing' or 'tuning'.\n"
              << "  -p  --parallel-tuning        Set tuning logic. Can be 'split' or 'sequential'. Default adapts to dataset size.\n"
              << "  -i  --path1                  First input path supplied, may be interpreted as training path or testing path.\n"
              << "  -I  --path2                  Second input path supplied, in tuning logic is interpreted as validation.\n"
              << "  -t  --target-column          Index of the target column.\n"
              << "  -c  --columns                Number of columns in the dataset.\n"
              << "  -r  --row1                   Number of rows in the first supplied dataset.\n"
              << "  -R  --row2                   Number of rows in the second supplied dataset.\n"
              << "  -H  --hyperparameters-path   Path to the hyperparameters file.\n"
              << "  -s  --svm-path               Path to the SVM file.\n"
              << "  -S  --save-dir-path          Folder path for saving SVM files.\n"
              << "  -M  --tuning-table-dir-path  Folder path for saving tuning table files.\n"
              << "  -k  --kernel                 Kernel type, may be 'l' (linear), 'p' (polynomial), 'r' (rbf) or 's' (sigmoid).\n"
              << "  -C  --cost                   Cost parameter.\n"
              << "  -g  --gamma                  Gamma parameter.\n"
              << "  -O  --coef0                  Coef0 parameter.\n"
              << "  -d  --degree                 Degree parameter.\n"
              << "  -T  --learning_rate          Learning rate parameter.\n"
              << "  -E  --eps                    Epsilon parameter.\n"
              << "  -L  --limit                  Limit parameter.\n"
              << "  -v  --verbose                Print verbose messages.\n"
              << std::endl;
    exit(0);
}

#endif

int main(int argc, char *argv[]) {

    /* Initialize MPI ------------------------------------ */

    MPI_Init(&argc, &argv);

    // Get the rank of the process
    int process_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    /* --------------------------------------------------- */

    bool performance_checks = read_env_var("PERFORMANCE_CHECKS").at(0) == 'T';
    int control = 0;

    if (performance_checks) {

        MPI_Barrier(MPI_COMM_WORLD);

        if (process_rank == MASTER_PROCESS) {
#if SHOW_LOGTIME
            logtime();
#endif
            std::cout << "This step is done by process 0 only to benchmark the speed of for loops\n";
            double start = MPI_Wtime(), end = 0;
            for (int i = 0; i < 1000; i++) {}
            end = MPI_Wtime();
#if SHOW_LOGTIME
            logtime();
#endif
            std::cout << "On this platform, a for loop cycle alone takes an average of " << (end - start) / 1000
                      << "\n\n";

        }

        MPI_Barrier(MPI_COMM_WORLD);

    }

    std::string filepath_training;
    std::string filepath_validation;
    std::string filepath_testing;
    std::string filepath_svm;

    std::string hparameters_path;
    std::string save_svm_dir_path;
    std::string save_tune_dir_path;

    bool defaul = true;

#if CLI_ARGS
// ----------------------- deal with cli args --------------------------------------------
    int next_option = 0;

    /* A string listing valid short options letters. */
    const char *const short_options = "hl:i:I:t:c:r:R:H:s:S:M:k:C:g:O:d:T:E:L:v";

    /* An array describing valid long options.  */
    const struct option long_options[] = {
            {"help",                  0, nullptr, 'h'},
            {"logic",                 1, nullptr, 'l'},
            {"parallel-tuning",       2, nullptr, 'p'},
            {"path1",                 1, nullptr, 'i'},
            {"path2",                 2, nullptr, 'I'},
            {"target-column",         1, nullptr, 't'},
            {"columns",               1, nullptr, 'c'},
            {"row1",                  1, nullptr, 'r'},
            {"row2",                  2, nullptr, 'R'},
            {"hyperparameters-path",  2, nullptr, 'H'},
            {"svm-path",              2, nullptr, 's'},
            {"save-dir-path",         2, nullptr, 'S'},
            {"tuning-table-dir-path", 2, nullptr, 'M'},
            {"kernel",                2, nullptr, 'k'},
            {"cost",                  2, nullptr, 'C'},
            {"gamma",                 2, nullptr, 'g'},
            {"coef0",                 2, nullptr, 'O'},
            {"degree",                2, nullptr, 'd'},
            {"learning_rate",         2, nullptr, 'T'},
            {"eps",                   2, nullptr, 'E'},
            {"limit",                 2, nullptr, 'L'},
            {"verbose",               0, nullptr, 'v'},
            {nullptr,                 0, nullptr, 0}
    };

    /**
     * Parameters initialization
     */

    bool tuning_logic = true;
    bool logic_set_flag = false;

    std::string p1;
    std::string p2;


    int rows_t = 0;
    int rows_v = 0;
    int target_column = 0;
    int columns = 0;


    char ker_type = DEFAULT_KERNEL;
    double Cost = DEFAULT_COST;
    double gamma = DEFAULT_GAMMA;
    double coef0 = DEFAULT_INTERCEPT;
    double degree = DEFAULT_DEGREE;
    double lr = DEFAULT_LEARNING_RATE;
    double limit = DEFAULT_LEARNING_RATE;
    double eps = DEFAULT_EPS;

    bool verbose = false;
    flag = undefined;

    std::string program_name = argv[0];


    /**
     * CLI arguments parsing
     */


    // ---------- assignments -----------
    do {

        next_option = getopt_long(argc, argv, short_options, long_options, nullptr);

        switch (next_option) {

            case 'h': {
                /* -h or --help */
                print_usage(program_name);
                break;
            }
            case 'l': {
                /* -l or --logic */
                if (strcmp(optarg, "training") == 0) {
                    if (process_rank == MASTER_PROCESS) {
#if SHOW_LOGTIME
                        logtime();
#endif
                        std::cout << "[INFO] Logic set to training." << std::endl;
                    }
                    flag = training;
                } else if (strcmp(optarg, "testing") == 0) {
                    flag = testing;
                    if (process_rank == MASTER_PROCESS) {
#if SHOW_LOGTIME
                        logtime();
#endif
                        std::cout << "[INFO] Logic set to testing." << std::endl;
                    }
                } else if (strcmp(optarg, "tuning") == 0) {
                    flag = tuning;
                    if (process_rank == MASTER_PROCESS) {
#if SHOW_LOGTIME
                        logtime();
#endif
                        std::cout << "[INFO] Logic set to tuning." << std::endl;
                    }
                } else {
                    if (process_rank == MASTER_PROCESS) {
#if SHOW_LOGTIME
                        logtime();
#endif
                        std::cout << "[ERROR] Invalid logic argument, please use training, testing or tuning.\n";
                    }
                    exit(1);
                }
                break;
            }

            case 'p': {   /* -p or --parallel-tuning */

                if (strcmp(optarg, "sequential") == 0) {
                    if (process_rank == MASTER_PROCESS) {
#if SHOW_LOGTIME
                        logtime();
#endif
                        std::cout
                                << "[INFO] Tuning logic set to sequential. All processes will go through sequential tuning with parallel training.\n";
                    }
                    tuning_logic = false;
                    logic_set_flag = true;
                } else if (strcmp(optarg, "split") == 0) {
                    if (process_rank == MASTER_PROCESS) {
#if SHOW_LOGTIME
                        logtime();
#endif
                        std::cout
                                << "[INFO] Tuning logic set to split. Processes will split the available combination.\n";
                    }
                    tuning_logic = true;
                    logic_set_flag = true;
                } else {
                    if (process_rank == MASTER_PROCESS) {
#if SHOW_LOGTIME
                        logtime();
#endif
                        std::cout
                                << "[INFO] Tuning logic has not been set. Processes will adapt with respect to dataset size.\n";
                    }

                }
                break;
            }

            case 'i': {  /* -i or --path1 */
                p1 = optarg;
                break;
            }

            case 'I': {
                p2 = optarg;
                break;
            }

            case 't': {
                /* -t or --target_column */
                target_column = std::atoi(optarg);
                if (process_rank == MASTER_PROCESS) {
#if SHOW_LOGTIME
                    logtime();
#endif
                    std::cout << "[INFO] Target column set to " << target_column << std::endl;
                }
                break;
            }

            case 'c': {  /* -c or --columns */
                columns = std::atoi(optarg);
                if (process_rank == MASTER_PROCESS) {
#if SHOW_LOGTIME
                    logtime();
#endif
                    std::cout << "[INFO] Number of columns set to " << columns << std::endl;
                }
                break;
            }

            case 'r': {
                rows_t = std::atoi(optarg);
                break;
            } /* -r or --row1 */

            case 'R': {   /* -R or --row2 */
                rows_v = std::atoi(optarg);
                break;
            }

            case 'H': {   /* -H or --hparameters_path */
                hparameters_path = optarg;
                break;
            }

            case 's': {  /* -s or --svm-path */
                char *tmp = optarg;
                if (tmp) {
                    filepath_svm = optarg;
                }
                if (process_rank == MASTER_PROCESS) {
#if SHOW_LOGTIME
                    logtime();
#endif
                    std::cout << "[INFO] SVM file path set to " << filepath_svm << std::endl;
                }
                break;
            }

            case 'S': {  /* -s or --svm_path */
                save_svm_dir_path = optarg;
                if (process_rank == MASTER_PROCESS) {
#if SHOW_LOGTIME
                    logtime();
#endif
                    std::cout << "[INFO] SVM save file path set to " << filepath_svm << std::endl;
                }
                break;
            }

            case 'M': {  /* -s or --svm_path */
                save_tune_dir_path = optarg;
                if (process_rank == MASTER_PROCESS) {
#if SHOW_LOGTIME
                    logtime();
#endif
                    std::cout << "[INFO] Tuning table file path set to " << filepath_svm << std::endl;
                }
                break;
            }

            case 'k': {  /* -k or --kernel */
                ker_type = *optarg;
                if (process_rank == MASTER_PROCESS) {
#if SHOW_LOGTIME
                    logtime();
#endif
                    std::cout << "[INFO] Kernel type set to `" << ker_type << "`" << std::endl;
                }
                break;
            }

            case 'C': { /* -C or --cost */
                Cost = std::stod(optarg);
                if (process_rank == MASTER_PROCESS) {
#if SHOW_LOGTIME
                    logtime();
#endif
                    std::cout << "[INFO] Cost parameter set to " << Cost << std::endl;
                }
                break;
            }

            case 'g': { /* -g or --gamma */
                gamma = std::stod(optarg);
                if (process_rank == MASTER_PROCESS) {
#if SHOW_LOGTIME
                    logtime();
#endif
                    std::cout << "[INFO] Gamma parameter set to " << gamma << std::endl;
                }
                break;
            }

            case 'O': { /* -O or --coef0 */
                coef0 = std::stod(optarg);
                if (process_rank == MASTER_PROCESS) {
#if SHOW_LOGTIME
                    logtime();
#endif
                    std::cout << "[INFO] Coef0 parameter set to " << coef0 << std::endl;
                }
                break;
            }

            case 'd': { /* -d or --degree */
                degree = std::stoi(optarg);
                if (process_rank == MASTER_PROCESS) {
#if SHOW_LOGTIME
                    logtime();
#endif
                    std::cout << "[INFO] Degree parameter set to " << degree << std::endl;
                }
                break;
            }

            case 'T': {   /* -T or --learning_rate */
                lr = std::stod(optarg);
#if SHOW_LOGTIME
                logtime();
#endif
                std::cout << "[INFO] Learning rate set to " << lr << std::endl;
                break;
            }

            case 'E': {  /* -E or --eps */
                eps = std::stod(optarg);
                if (process_rank == MASTER_PROCESS) {
#if SHOW_LOGTIME
                    logtime();
#endif
                    std::cout << "[INFO] Epsilon set to " << eps << std::endl;
                }
                break;
            }

            case 'L': {  /* -L or --limit */
                limit = std::stod(optarg);
                if (process_rank == MASTER_PROCESS) {
#if SHOW_LOGTIME
                    logtime();
#endif
                    std::cout << "[INFO] Limit value set to " << limit << std::endl;
                }
                break;
            }

            case 'v': { /* -v or --verbose */
                verbose = true;
                break;
            }

            case '?': {  /* The user specified an invalid option */
                if (process_rank == MASTER_PROCESS) {
#if SHOW_LOGTIME
                    logtime();
#endif
                    std::cout << "\n[WARN] You entered an invalid option." << std::endl;
                }
                // print_usage(1);
            } // ?

            case -1: {   /* Done with options */
                break;
            }

            default: {
                /* Something else unexpected */
                exit(1);
            }

        }

    } while (next_option != -1);


    //----------- checks ----------------

    if (!p1.empty()) {
        if ((flag == training) || (flag == tuning)) {
            filepath_training = std::move(p1);
            if (process_rank == MASTER_PROCESS) {
#if SHOW_LOGTIME
                logtime();
#endif
                std::cout << "[INFO] Training file path set to " << filepath_training << std::endl;
            }
        } else if (flag == testing) {
            filepath_testing = std::move(p1);
            if (process_rank == MASTER_PROCESS) {
#if SHOW_LOGTIME
                logtime();
#endif
                std::cout << "[INFO] Testing file path set to " << filepath_testing << std::endl;
            }
        } else {
            if (process_rank == MASTER_PROCESS) {
#if SHOW_LOGTIME
                logtime();
#endif
                std::cout << "[ERROR] Program logic (tuning, training, testing) was not defined.\n";
            }
            exit(1);
        }
    } else {
        if (process_rank == MASTER_PROCESS) {
#if SHOW_LOGTIME
            logtime();
#endif
            std::cout << "[ERROR] Argument path1 was not supplied.\n";
        }
        exit(1);
    }

    if (!p2.empty()) {
        if (flag != tuning) {
#if SHOW_LOGTIME
            logtime();
#endif
            std::cout << "\n[WARN] You entered path2, but logic was not set to tuning." << std::endl;
        } else {
            filepath_validation = std::move(p2);
        }
    } else {
        if (flag == tuning) {
#if SHOW_LOGTIME
            logtime();
#endif
            std::cout << "\n[ERROR] Path2 was not entered, but logic was set to tuning." << std::endl;
            exit(1);
        }
    }

    if (!logic_set_flag) {
        tuning_logic = rows_t > columns;
        if (process_rank == MASTER_PROCESS) {
            std::cout
                    << "\n[WARN] Tuning logic was not supplied, therefore the program will adapt based on the supplied datasets\n";
        }
    }

    if ((rows_v != 0) && flag != tuning) {
        if (process_rank == MASTER_PROCESS) {
#if SHOW_LOGTIME
            logtime();
#endif
            std::cout << "[WARN] Rows for validation dataset were passed, but the programi is not on tuning mode.\n"
                      << std::endl;
        }

    }

    if (target_column == 0 || columns == 0 || rows_t == 0) {
        if (process_rank == MASTER_PROCESS) {
#if SHOW_LOGTIME
            logtime();
#endif
            std::cout << "\n[ERROR] A lenght was not entered\nRows=" << rows_t << "\ncolumns=" << columns
                      << "\ntarget column=" << target_column << std::endl;
        }
        exit(1);
    }

    if (flag == undefined) {
        if (process_rank == MASTER_PROCESS) {
#if SHOW_LOGTIME
            logtime();
#endif
            std::cout << "\n[ERROR] Program logic was not defined " << std::endl;
        }
        exit(1);
    }

    if (flag == testing && filepath_svm.empty()) {
        if (process_rank == MASTER_PROCESS) {
#if SHOW_LOGTIME
            logtime();
#endif
            std::cout << "\n[ERROR] SVM path was not provided! " << std::endl;
        }
        exit(1);
    }

    if (process_rank == MASTER_PROCESS) {
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

    bool tuning_logic = true;

    /* Antonio */
     // filepath_training = "hpc2022/data/iris_train.csv";
     // filepath_validation = "hpc2022/data/iris_validation.csv";
     // save_svm_dir_path = "hpc2022/saved_svm/";


    /* Maurizio */
    filepath_training = "/home/dmmp/Documents/GitHub/hpc2022/data/iris_train.csv";
    filepath_validation = "/home/dmmp/Documents/GitHub/hpc2022/data/iris_validation.csv";
    save_svm_dir_path = "hpc2022/saved_svm";


    std::string filepath_hyper_parameters = "hpc2022/data/hyperparameters.csv"; // TODO: implement
    filepath_svm = "hpc2022/saved_svm/sigmoids_C0.500000_G0.010000_O0.000000.svm";

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


    /**
     * Program startup
     */

    auto *time_checks = (double *) calloc(NUMBER_OF_PERFORMANCE_CHECKS, sizeof(double));
    int time_iterator = 0;
    if (performance_checks) {

        MPI_Barrier(MPI_COMM_WORLD);
        *(time_checks + time_iterator) = MPI_Wtime(); // start


        if (process_rank == MASTER_PROCESS) {
#if SHOW_LOGTIME
            logtime();
#endif
            std::cout << "[INFO] Program starts at time " << *(time_checks + time_iterator) << "\n" << std::endl;
        }
        ++time_iterator;
    }

    /**
     * Switch block begins
     */

    switch (flag) {


        /// Training case

        case train_flag::training : {

// -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

            if (performance_checks) {

                MPI_Barrier(MPI_COMM_WORLD);
                *(time_checks + time_iterator) = MPI_Wtime(); // start

                if (process_rank == MASTER_PROCESS) {
#if SHOW_LOGTIME
                    logtime();
#endif
                    std::cout << "[INFO] Reading training dataset starts at time " << *(time_checks + time_iterator)
                              << std::endl;
                }
                ++time_iterator;
            }


            Dataset df_train = read_dataset(filepath_training, rows_t, columns, target_column);

            if (performance_checks) {

                MPI_Barrier(MPI_COMM_WORLD);
                *(time_checks + time_iterator) = MPI_Wtime(); // start

                if (process_rank == MASTER_PROCESS) {

#if SHOW_LOGTIME
                    logtime();
#endif
                    std::cout << "[INFO] It took " << *(time_checks + time_iterator) - *(time_checks + time_iterator - 1)
                              << " seconds\n" << std::endl;

#if SHOW_LOGTIME
                    logtime();
#endif
                    std::cout << "[INFO] svm preparation starts at time " << *(time_checks + time_iterator) << std::endl;
                }
                ++time_iterator;
            }


            Kernel_SVM svm;

            svm.verbose = verbose;
            set_kernel_function(&svm, ker_type);

            double params[4] = {Cost, gamma, coef0, degree};


            if (performance_checks) {

                MPI_Barrier(MPI_COMM_WORLD);
                *(time_checks + time_iterator) = MPI_Wtime(); // start

                if (process_rank == MASTER_PROCESS) {

#if SHOW_LOGTIME
                    logtime();
#endif
                    std::cout << "[INFO] It took " << *(time_checks + time_iterator) - *(time_checks + time_iterator - 1)
                              << " seconds\n" << std::endl;
#if SHOW_LOGTIME
                    logtime();
#endif
                    std::cout << "[INFO] Training starts at time " << *(time_checks + time_iterator) << std::endl;
                }
                ++time_iterator;
            }

            parallel_train(df_train, &svm, params, lr, limit, MASTER_PROCESS, world_size, true, save_svm_dir_path, 0,
                           eps);

            if (performance_checks) {

                MPI_Barrier(MPI_COMM_WORLD);
                *(time_checks + time_iterator) = MPI_Wtime(); // start

                if (process_rank == MASTER_PROCESS) {

#if SHOW_LOGTIME
                    logtime();
#endif
                    std::cout << "[INFO] It took " << *(time_checks + time_iterator) - *(time_checks + time_iterator - 1)
                              << " seconds\n" << std::endl;
                }
                ++time_iterator;
            }

            break;
        }

            /// Testing case

        case train_flag::testing: {

/* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- */
            if (performance_checks) {

                MPI_Barrier(MPI_COMM_WORLD);
                *(time_checks + time_iterator) = MPI_Wtime(); // start

                if (process_rank == MASTER_PROCESS) {
#if SHOW_LOGTIME
                    logtime();
#endif
                    std::cout << "[INFO] Reading testing dataset starts at time " << *(time_checks + time_iterator)
                              << std::endl;
                }
                ++time_iterator;
            }
            Dataset df_test = read_dataset(filepath_testing, rows_t, columns, target_column);
            if (performance_checks) {

                MPI_Barrier(MPI_COMM_WORLD);
                *(time_checks + time_iterator) = MPI_Wtime(); // start

                if (process_rank == MASTER_PROCESS) {

#if SHOW_LOGTIME
                    logtime();
#endif
                    std::cout << "[INFO] It took " << *(time_checks + time_iterator) - *(time_checks + time_iterator - 1)
                              << " seconds\n" << std::endl;
                }
                ++time_iterator;
            }

            Kernel_SVM svm;

            if (performance_checks) {

                MPI_Barrier(MPI_COMM_WORLD);
                *(time_checks + time_iterator) = MPI_Wtime(); // start

                if (process_rank == MASTER_PROCESS) {
#if SHOW_LOGTIME
                    logtime();
#endif
                    std::cout << "[INFO] Reading svm file starts at time " << *(time_checks + time_iterator) << std::endl;
                }
                ++time_iterator;
            }
            read_svm(&svm, filepath_svm);
            if (performance_checks) {

                MPI_Barrier(MPI_COMM_WORLD);
                *(time_checks + time_iterator) = MPI_Wtime(); // start

                if (process_rank == MASTER_PROCESS) {

#if SHOW_LOGTIME
                    logtime();
#endif
                    std::cout << "[INFO] It took " << *(time_checks + time_iterator) - *(time_checks + time_iterator - 1)
                              << " seconds\n" << std::endl;
                }
                ++time_iterator;
            }

            svm.verbose = true;

            if (performance_checks) {

                MPI_Barrier(MPI_COMM_WORLD);
                *(time_checks + time_iterator) = MPI_Wtime(); // start

                if (process_rank == MASTER_PROCESS) {
#if SHOW_LOGTIME
                    logtime();
#endif
                    std::cout << "[INFO] Parallel test starts at time " << *(time_checks + time_iterator) << std::endl;
                }
                ++time_iterator;
            }
            parallel_test(df_test, &svm, MASTER_PROCESS, world_size);
            if (performance_checks) {

                MPI_Barrier(MPI_COMM_WORLD);
                *(time_checks + time_iterator) = MPI_Wtime(); // start

                if (process_rank == MASTER_PROCESS) {

#if SHOW_LOGTIME
                    logtime();
#endif
                    std::cout << "[INFO] It took " << *(time_checks + time_iterator) - *(time_checks + time_iterator - 1)
                              << " seconds\n" << std::endl;
                }
                ++time_iterator;
            }
            break;

        }

            /// Tuning case

        case train_flag::tuning: {

/* ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- */

            if (process_rank == MASTER_PROCESS) {
#if SHOW_LOGTIME
                logtime();
#endif
                std::cout << "[INFO] Training Dataset filepath: " << filepath_training << std::endl;

#if SHOW_LOGTIME
                logtime();
#endif
                std::cout << "[INFO] The dataset has " << rows_t << " rows and " << columns << " columns." << std::endl;

#if SHOW_LOGTIME
                logtime();
#endif
                std::cout << "[INFO] Validation Dataset filepath: " << filepath_validation << std::endl;

#if SHOW_LOGTIME
                logtime();
#endif
                std::cout << "[INFO] The dataset has " << rows_v << " rows and " << columns << " columns." << std::endl;
            }


#if CLI_ARGS
            double *cost_array;
            double *gamma_array;
            double *coef0_array;
            double *degree_array;

            int cost_array_size, gamma_array_size, coef0_array_size, degree_array_size;
            cost_array = (double *) calloc(sizeof(double), MAX_HP_VALUES);
            gamma_array = (double *) calloc(sizeof(double), MAX_HP_VALUES);
            coef0_array = (double *) calloc(sizeof(double), MAX_HP_VALUES);
            degree_array = (double *) calloc(sizeof(double), MAX_HP_VALUES);

            if (!hparameters_path.empty()) {


                cost_array_size = 0;
                gamma_array_size = 0;
                coef0_array_size = 0;
                degree_array_size = 0;

                read_hyperparameters(hparameters_path,
                                     cost_array, cost_array_size,
                                     gamma_array, gamma_array_size,
                                     coef0_array, coef0_array_size,
                                     degree_array, degree_array_size);

                cost_array = (double *) realloc(cost_array, cost_array_size * sizeof(double));
                gamma_array = (double *) realloc(gamma_array, gamma_array_size * sizeof(double));
                coef0_array = (double *) realloc(coef0_array, coef0_array_size * sizeof(double));
                degree_array = (double *) realloc(degree_array, degree_array_size * sizeof(double));
                defaul = false;

            } else {
                if (process_rank == MASTER_PROCESS) {
#if SHOW_LOGTIME
                    logtime();
#endif
                    std::cout << "[WARN] Tuning file was not supplied, using default values\n";
                }

                cost_array_size = DEFAULT_COST_SIZE;
                gamma_array_size = DEFAULT_GAMMA_SIZE;
                coef0_array_size = DEFAULT_COEF0_SIZE;
                degree_array_size = DEFAULT_DEGREE_SIZE;

                cost_array = DEFAULT_COST_ARRAY;
                gamma_array = DEFAULT_GAMMA_ARRAY;
                coef0_array = DEFAULT_COEF0_ARRAY;
                degree_array = DEFAULT_DEGREE_ARRAY;

            }


            if (process_rank == MASTER_PROCESS) {

                std::cout << "\n" << std::endl;

#if SHOW_LOGTIME
                logtime();
#endif
                std::cout << "[INFO] Cost array size: " << cost_array_size << std::endl;
#if SHOW_LOGTIME
                logtime();
#endif
                std::cout << "[INFO] Cost array:";

                for (int i = 0; i < cost_array_size; i++) {
                    std::cout << cost_array[i] << " ";
                }
                std::cout << std::endl;

#if SHOW_LOGTIME
                logtime();
#endif
                std::cout << "[INFO] Gamma array size: " << gamma_array_size << std::endl;
#if SHOW_LOGTIME
                logtime();
#endif
                std::cout << "[INFO] Gamma array:";

                for (int i = 0; i < gamma_array_size; i++) {
                    std::cout << gamma_array[i] << " ";
                }
                std::cout << std::endl;

#if SHOW_LOGTIME
                logtime();
#endif
                std::cout << "[INFO] Coef0 array size: " << coef0_array_size << std::endl;
#if SHOW_LOGTIME
                logtime();
#endif
                std::cout << "[INFO] Coef0 array:";

                for (int i = 0; i < coef0_array_size; i++) {
                    std::cout << coef0_array[i] << " ";
                }
                std::cout << std::endl;


#if SHOW_LOGTIME
                logtime();
#endif
                std::cout << "[INFO] Degree array size: " << degree_array_size << std::endl;
#if SHOW_LOGTIME
                logtime();
#endif
                std::cout << "[INFO] Degree array: ";

                for (int i = 0; i < degree_array_size; i++) {
                    std::cout << degree_array[i] << " ";
                }
                std::cout << std::endl;

            }

#else

            int cost_array_size = DEFAULT_COST_SIZE, gamma_array_size = DEFAULT_GAMMA_SIZE, coef0_array_size = DEFAULT_COEF0_SIZE, degree_array_size = DEFAULT_DEGREE_SIZE;
            double* cost_array = DEFAULT_COST_ARRAY;
            double* gamma_array = DEFAULT_GAMMA_ARRAY;
            double* coef0_array = DEFAULT_COEF0_ARRAY;
            double* degree_array = DEFAULT_DEGREE_ARRAY;

#endif
            if (performance_checks) {

                MPI_Barrier(MPI_COMM_WORLD);
                *(time_checks + time_iterator) = MPI_Wtime();

                if (process_rank == MASTER_PROCESS) {

                    std::cout << "\n" << std::endl;

#if SHOW_LOGTIME
                    logtime();
#endif
                    std::cout << "[INFO] Allocating inital vectors took "
                              << *(time_checks + time_iterator) - *(time_checks + time_iterator - 1) << " seconds\n"
                              << std::endl;
                }


                if (process_rank == MASTER_PROCESS) {
#if SHOW_LOGTIME
                    logtime();
#endif
                    std::cout << "[INFO] Training dataset read starts at time " << *(time_checks + time_iterator) << "\n"
                              << std::endl;
                }
                ++time_iterator; // start tr dataset read
            }


            Dataset df_train = read_dataset(filepath_training, rows_t, columns, target_column);

            if (performance_checks) {

                MPI_Barrier(MPI_COMM_WORLD);
                *(time_checks + time_iterator) = MPI_Wtime(); // end of allocations

                if (process_rank == MASTER_PROCESS) {
#if SHOW_LOGTIME
                    logtime();
#endif
                    std::cout << "[INFO] Reading training dataset took a total of "
                              << *(time_checks + time_iterator) - *(time_checks + time_iterator - 1) << " seconds\n"
                              << std::endl;
                }

                if (process_rank == MASTER_PROCESS) {
#if SHOW_LOGTIME
                    logtime();
#endif
                    std::cout << "[INFO] Validation dataset read starts at time " << *(time_checks + time_iterator) << "\n"
                              << std::endl;
                }
                ++time_iterator; // start val dataset read
            }

            Dataset df_validation = read_dataset(filepath_validation, rows_v, columns, target_column);

            if (performance_checks) {

                MPI_Barrier(MPI_COMM_WORLD);
                *(time_checks + time_iterator) = MPI_Wtime(); // end of allocations

                if (process_rank == MASTER_PROCESS) {
#if SHOW_LOGTIME
                    logtime();
#endif
                    std::cout << "[INFO] Reading validation dataset took "
                              << *(time_checks + time_iterator) - *(time_checks + time_iterator - 1) << " seconds\n"
                              << std::endl;
                }
                if (process_rank == MASTER_PROCESS) {
#if SHOW_LOGTIME
                    logtime();
#endif
                    std::cout << "[INFO] Memory allocation for tuning starts at time " << *(time_checks + time_iterator)
                              << "\n"
                              << std::endl;
                }
                ++time_iterator;
            }


            // TODO: refactor?

            int linear_rows = cost_array_size;
            int radial_rows = cost_array_size * gamma_array_size;
            int sigmoid_rows = cost_array_size * gamma_array_size * coef0_array_size;
            int polynomial_rows = cost_array_size * gamma_array_size * coef0_array_size * degree_array_size;

            int tuning_table_rows = linear_rows + radial_rows + sigmoid_rows + polynomial_rows;
            int tuning_table_columns = NUMBER_OF_HYPER_PARAMETERS +
                                       1/* accuracy*/ +
                                       1/*class 1 accuracy*/ +
                                       1/*class 2 accuracy*/; // NB: type of kernel will be printed separately
            if (process_rank == MASTER_PROCESS) {
#if SHOW_LOGTIME
                logtime();
#endif
                std::cout << "[INFO] There are a total of " << IMPLEMENTED_KERNELS << " kernels to tune." << std::endl;
#if SHOW_LOGTIME
                logtime();
#endif
                std::cout << "[INFO] Tuning will use: " <<
                          cost_array_size << " different values for costs, " <<
                          gamma_array_size << " for gamma, " <<
                          coef0_array_size << " for intercepts, " <<
                          degree_array_size << " for exponential degrees" << std::endl;
#if SHOW_LOGTIME
                logtime();
#endif
                std::cout << "[INFO] For a total of " << tuning_table_rows << " combinations" << std::endl;
            }

            auto *final_tuning_table = (double *) calloc(tuning_table_rows * tuning_table_columns,
                                                         sizeof(double)); // matrix

            auto *local_tuning_table = (double *) calloc(tuning_table_rows * tuning_table_columns,
                                                         sizeof(double)); // matrix

            if (tuning_logic) {
                // split

#if PERFORMANCE_CHECK

                MPI_Barrier(MPI_COMM_WORLD);
                *(time_checks + time_iterator) = MPI_Wtime(); // end of allocations

                if (process_rank == MASTER_PROCESS) {
#if SHOW_LOGTIME
                    logtime();
#endif
                    std::cout << "[INFO] Latest memory allocation ends at time " << *(time_checks + time_iterator)
                              << std::endl;
#if SHOW_LOGTIME
                    logtime();
#endif
                    std::cout << "[INFO] It took " << *(time_checks + time_iterator) - *(time_checks + time_iterator - 1)
                              << " seconds\n" << std::endl;

                }
                ++time_iterator; // start linear tuning

#endif

                //linear
                if (process_rank == MASTER_PROCESS) {
#if SHOW_LOGTIME
                    logtime();
#endif
                    std::cout << "[INFO] Starting linear tuning" << std::endl;
                }
                tune_linear(&df_train, &df_validation, cost_array, cost_array_size, local_tuning_table, 0,
                            tuning_table_columns, MASTER_PROCESS, world_size, lr, limit, eps, verbose);
#if SHOW_LOGTIME
                logtime();
#endif
                std::cout << "[INFO] Process " << process_rank << " has finished linear tuning" << std::endl;

                //radial
                if (performance_checks) {

                    MPI_Barrier(MPI_COMM_WORLD);
                    *(time_checks + time_iterator) = MPI_Wtime();

                    if (process_rank == MASTER_PROCESS) {
#if SHOW_LOGTIME
                        logtime();
#endif
                        std::cout << "[INFO] linear tuning ends at time " << *(time_checks + time_iterator) << std::endl;
#if SHOW_LOGTIME
                        logtime();
#endif
                        std::cout << "[INFO] It took "
                                  << *(time_checks + time_iterator) - *(time_checks + time_iterator - 1)
                                  << "seconds\n" << std::endl;
                    }
                    ++time_iterator; // start radial tuning
                }
                if (process_rank == MASTER_PROCESS) {
#if SHOW_LOGTIME
                    logtime();
#endif
                    std::cout << "[INFO] Starting radial tuning" << std::endl;
                }
                tune_radial(&df_train, &df_validation, cost_array, cost_array_size, gamma_array, gamma_array_size,
                            local_tuning_table, linear_rows, tuning_table_columns, MASTER_PROCESS, world_size, lr,
                            limit, eps, verbose);
#if SHOW_LOGTIME
                logtime();
#endif
                std::cout << "[INFO] Process " << process_rank << " has finished radial tuning" << std::endl;

                //sigmoid
                if (performance_checks) {

                    MPI_Barrier(MPI_COMM_WORLD);
                    *(time_checks + time_iterator) = MPI_Wtime();

                    if (process_rank == MASTER_PROCESS) {
#if SHOW_LOGTIME
                        logtime();
#endif
                        std::cout << "[INFO] radial tuning ends at time " << *(time_checks + time_iterator) << std::endl;
#if SHOW_LOGTIME
                        logtime();
#endif
                        std::cout << "[INFO] It took "
                                  << *(time_checks + time_iterator) - *(time_checks + time_iterator - 1)
                                  << "seconds\n" << std::endl;
                    }
                    ++time_iterator; // start sigmoid tuning
                }
                if (process_rank == MASTER_PROCESS) {
#if SHOW_LOGTIME
                    logtime();
#endif
                    std::cout << "[INFO] Starting sigmoid tuning" << std::endl;
                }
                tune_sigmoid(&df_train, &df_validation, cost_array, cost_array_size, gamma_array, gamma_array_size,
                             coef0_array, coef0_array_size, local_tuning_table, linear_rows + radial_rows,
                             tuning_table_columns, MASTER_PROCESS, world_size);
#if SHOW_LOGTIME
                logtime();
#endif
                std::cout << "[INFO] Process " << process_rank << " has finished sigmoid tuning" << std::endl;

                //polynomial
                if (performance_checks) {

                    MPI_Barrier(MPI_COMM_WORLD);
                    *(time_checks + time_iterator) = MPI_Wtime();

                    if (process_rank == MASTER_PROCESS) {
#if SHOW_LOGTIME
                        logtime();
#endif
                        std::cout << "[INFO] sigmoid tuning ends at time " << *(time_checks + time_iterator) << std::endl;
#if SHOW_LOGTIME
                        logtime();
#endif
                        std::cout << "[INFO] It took "
                                  << *(time_checks + time_iterator) - *(time_checks + time_iterator - 1)
                                  << "seconds\n" << std::endl;
                    }
                    ++time_iterator; // start polynomial tuning
                }
                if (process_rank == MASTER_PROCESS) {
#if SHOW_LOGTIME
                    logtime();
#endif
                    std::cout << "[INFO] Starting polynomial tuning" << std::endl;
                }
                tune_polynomial(&df_train, &df_validation, cost_array, cost_array_size, gamma_array, gamma_array_size,
                                coef0_array, coef0_array_size, degree_array, degree_array_size, local_tuning_table,
                                linear_rows + radial_rows + sigmoid_rows, tuning_table_columns, MASTER_PROCESS,
                                world_size);
#if SHOW_LOGTIME
                logtime();
#endif
                std::cout << "[INFO] Process " << process_rank << " has finished polynomial tuning" << std::endl;
                if (performance_checks) {

                    MPI_Barrier
                            (MPI_COMM_WORLD);
                    *(time_checks + time_iterator) = MPI_Wtime();

                    if (process_rank == MASTER_PROCESS) {
#if SHOW_LOGTIME
                        logtime();
#endif
                        std::cout << "[INFO] polynomial tuning ends at time " << *(time_checks + time_iterator)
                                  << std::endl;
#if SHOW_LOGTIME
                        logtime();
#endif
                        std::cout << "[INFO] It took "
                                  << *(time_checks + time_iterator) - *(time_checks + time_iterator - 1)
                                  << "seconds\n" << std::endl;
#if SHOW_LOGTIME
                        logtime();
#endif
                        std::cout << "[INFO] Starting reduce" << std::endl;
                    }
                    ++time_iterator; // start reduce
                }

                MPI_Reduce(local_tuning_table, final_tuning_table, (int) (tuning_table_rows * tuning_table_columns),
                           MPI_DOUBLE, MPI_SUM, MASTER_PROCESS, MPI_COMM_WORLD);


                if (performance_checks) {
                    MPI_Barrier(MPI_COMM_WORLD);
                    *(time_checks + time_iterator) = MPI_Wtime();

                    if (process_rank == MASTER_PROCESS) {
#if SHOW_LOGTIME
                        logtime();
#endif
                        std::cout << "[INFO] Reduce ends at time " << *(time_checks + time_iterator) << std::endl;
#if SHOW_LOGTIME
                        logtime();
#endif
                        std::cout << "[INFO] It took "
                                  << *(time_checks + time_iterator) - *(time_checks + time_iterator - 1)
                                  << "seconds\n" << std::endl;
                    }
                    ++time_iterator; // start calculating accuracy
                }

#if DEBUG_MAIN
                if(process_rank == MASTER_PROCESS){
                    print_matrix(final_tuning_table, tuning_table_rows, tuning_table_columns);
                }
#endif

            } else {  // sequential tuning, parallel training

                if (performance_checks) {

                    MPI_Barrier(MPI_COMM_WORLD);
                    *(time_checks + time_iterator) = MPI_Wtime(); // end of allocations

                    if (process_rank == MASTER_PROCESS) {
#if SHOW_LOGTIME
                        logtime();
#endif
                        std::cout << "[INFO] Latest memory allocation ends at time " << *(time_checks + time_iterator)
                                  << std::endl;
#if SHOW_LOGTIME
                        logtime();
#endif
                        std::cout << "[INFO] It took " << *(time_checks + time_iterator) - *(time_checks + time_iterator - 1)
                                  << " seconds\n" << std::endl;

                    }
                    ++time_iterator; // start linear tuning
                }


                //linear
                if (process_rank == MASTER_PROCESS) {
#if SHOW_LOGTIME
                    logtime();
#endif
                    std::cout << "[INFO] Starting linear tuning" << std::endl;
                }
                tune_linear2(&df_train, &df_validation, cost_array, cost_array_size, local_tuning_table, 0,
                             tuning_table_columns, MASTER_PROCESS, world_size, lr, limit, eps, verbose);
                if (process_rank == MASTER_PROCESS) {
#if SHOW_LOGTIME
                    logtime();
#endif
                    std::cout << "[INFO] Processes have finished linear tuning" << std::endl;
                }

                //radial
                if (performance_checks) {

                    MPI_Barrier(MPI_COMM_WORLD);
                    *(time_checks + time_iterator) = MPI_Wtime();

                    if (process_rank == MASTER_PROCESS) {
#if SHOW_LOGTIME
                        logtime();
#endif
                        std::cout << "[INFO] linear tuning ends at time " << *(time_checks + time_iterator) << std::endl;
#if SHOW_LOGTIME
                        logtime();
#endif
                        std::cout << "[INFO] It took "
                                  << *(time_checks + time_iterator) - *(time_checks + time_iterator - 1)
                                  << "seconds\n" << std::endl;
                    }
                    ++time_iterator; // start radial tuning
                }
                if (process_rank == MASTER_PROCESS) {
#if SHOW_LOGTIME
                    logtime();
#endif
                    std::cout << "[INFO] Starting radial tuning" << std::endl;
                }
                tune_radial2(&df_train, &df_validation, cost_array, cost_array_size, gamma_array, gamma_array_size,
                             local_tuning_table, linear_rows, tuning_table_columns, MASTER_PROCESS, world_size, lr,
                             limit, eps, verbose);
                if (process_rank == MASTER_PROCESS) {
#if SHOW_LOGTIME
                    logtime();
#endif
                    std::cout << "[INFO] Processes have finished linear tuning" << std::endl;
                }

                //sigmoid
                if (performance_checks) {
                    MPI_Barrier(MPI_COMM_WORLD);
                    *(time_checks + time_iterator) = MPI_Wtime();

                    if (process_rank == MASTER_PROCESS) {
#if SHOW_LOGTIME
                        logtime();
#endif
                        std::cout << "[INFO] radial tuning ends at time " << *(time_checks + time_iterator) << std::endl;
#if SHOW_LOGTIME
                        logtime();
#endif
                        std::cout << "[INFO] It took "
                                  << *(time_checks + time_iterator) - *(time_checks + time_iterator - 1)
                                  << "seconds\n" << std::endl;
                    }
                    ++time_iterator; // start sigmoid tuning
                }
                if (process_rank == MASTER_PROCESS) {
#if SHOW_LOGTIME
                    logtime();
#endif
                    std::cout << "[INFO] Starting sigmoid tuning" << std::endl;
                }
                tune_sigmoid2(&df_train, &df_validation, cost_array, cost_array_size, gamma_array, gamma_array_size,
                              coef0_array, coef0_array_size, local_tuning_table, linear_rows + radial_rows,
                              tuning_table_columns, MASTER_PROCESS, world_size);
                if (process_rank == MASTER_PROCESS) {
#if SHOW_LOGTIME
                    logtime();
#endif
                    std::cout << "[INFO] Processes have finished linear tuning" << std::endl;
                }

                //polynomial
                if (performance_checks) {

                    MPI_Barrier(MPI_COMM_WORLD);
                    *(time_checks + time_iterator) = MPI_Wtime();

                    if (process_rank == MASTER_PROCESS) {
#if SHOW_LOGTIME
                        logtime();
#endif
                        std::cout << "[INFO] sigmoid tuning ends at time " << *(time_checks + time_iterator) << std::endl;
#if SHOW_LOGTIME
                        logtime();
#endif
                        std::cout << "[INFO] It took "
                                  << *(time_checks + time_iterator) - *(time_checks + time_iterator - 1)
                                  << "seconds\n" << std::endl;
                    }
                    ++time_iterator; // start polynomial tuning
                }
                if (process_rank == MASTER_PROCESS) {
#if SHOW_LOGTIME
                    logtime();
#endif
                    std::cout << "[INFO] Starting polynomial tuning" << std::endl;
                }
                tune_polynomial2(&df_train, &df_validation, cost_array, cost_array_size, gamma_array, gamma_array_size,
                                 coef0_array, coef0_array_size, degree_array, degree_array_size, local_tuning_table,
                                 linear_rows + radial_rows + sigmoid_rows, tuning_table_columns, MASTER_PROCESS,
                                 world_size);
                if (process_rank == MASTER_PROCESS) {
#if SHOW_LOGTIME
                    logtime();
#endif
                    std::cout << "[INFO] Processes have finished linear tuning" << std::endl;
                }
                if (performance_checks) {

                    MPI_Barrier
                            (MPI_COMM_WORLD);
                    *(time_checks + time_iterator) = MPI_Wtime();

                    if (process_rank == MASTER_PROCESS) {
#if SHOW_LOGTIME
                        logtime();
#endif
                        std::cout << "[INFO] polynomial tuning ends at time " << *(time_checks + time_iterator)
                                  << std::endl;
#if SHOW_LOGTIME
                        logtime();
#endif
                        std::cout << "[INFO] It took "
                                  << *(time_checks + time_iterator) - *(time_checks + time_iterator - 1)
                                  << "seconds\n" << std::endl;
#if SHOW_LOGTIME
                        logtime();
#endif
                        std::cout << "[INFO] Starting reduce\n" << std::endl;
                    }
                    ++time_iterator; // start reduce
                }
                final_tuning_table = local_tuning_table;
                if (performance_checks) {

                    MPI_Barrier(MPI_COMM_WORLD);
                    *(time_checks + time_iterator) = MPI_Wtime();

                    if (process_rank == MASTER_PROCESS) {
#if SHOW_LOGTIME
                        logtime();
#endif
                        std::cout << "[INFO] Reduce ends at time " << *(time_checks + time_iterator) << std::endl;
#if SHOW_LOGTIME
                        logtime();
#endif
                        std::cout << "[INFO] It took "
                                  << *(time_checks + time_iterator) - *(time_checks + time_iterator - 1)
                                  << "seconds\n" << std::endl;
                    }
                    ++time_iterator; // start calculating accuracy
                }


#if DEBUG_MAIN
                if(process_rank == MASTER_PROCESS){
                    print_matrix(final_tuning_table, tuning_table_rows, tuning_table_columns);
                }
#endif
            }

            if (!defaul) {
                free(cost_array);
                cost_array = nullptr;
                free(gamma_array);
                gamma_array = nullptr;
                free(coef0_array);
                coef0_array = nullptr;
                free(degree_array);
                degree_array = nullptr;
            }

            if (process_rank == MASTER_PROCESS) {


                if (save_tune_dir_path.empty()) {
#if SHOW_LOGTIME
                    logtime();
#endif
                    std::cout
                            << "[WARN] Tuning save path was not passed, therefore the tuning table will be saved in the current directory"
                            << std::endl;
                }
#if SHOW_LOGTIME
                logtime();
#endif
                std::cout << "[INFO] Starting tuning table save" << std::endl;
                control += save_tuning_table(save_tune_dir_path, final_tuning_table, tuning_table_rows,
                                             tuning_table_columns, linear_rows, radial_rows, sigmoid_rows,
                                             polynomial_rows);
                if (control > 0) {
#if SHOW_LOGTIME
                    logtime();
#endif
                    std::cout << "[ERROR] Error while saving tuning table\n";
                }
// --------------------------------------------------- error is inside here ---------------------------------------------------------------------------------------------
                // if (performance_checks) {
////
                //     MPI_Barrier(MPI_COMM_WORLD);
                //     *(time_checks + time_iterator) = MPI_Wtime();
////
                //     if (process_rank == MASTER_PROCESS) {
                //         std::cout << "\n Saving tuning table ends at time " << *(time_checks + time_iterator) << std::endl;
                //         std::cout << "\n It took "
                //                   << *(time_checks + time_iterator) - *(time_checks + time_iterator - 1)
                //                   << "seconds\n" << std::endl;
                //     }
                //     ++time_iterator; // start getting best row
                // }

// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

                // todo: revise selection logic
                auto *accuracies = (double *) calloc(tuning_table_rows, sizeof(double));
                get_column(final_tuning_table, NUMBER_OF_HYPER_PARAMETERS, tuning_table_columns, tuning_table_rows,
                           accuracies);
                auto *m = std::max_element(accuracies, accuracies + tuning_table_rows);
                int row_index = (int) (m - accuracies);
                double max_accuracy = *m;

                std::cout << "\n" << std::endl;

#if SHOW_LOGTIME
                logtime();
#endif
                std::cout << "[INFO] Best Accuracy: " << max_accuracy << std::endl;
                auto *best_row = (double *) calloc(tuning_table_columns, sizeof(double));
                get_row(final_tuning_table, row_index, tuning_table_columns, best_row);
#if SHOW_LOGTIME
                logtime();
#endif
                std::cout << "[INFO] Best combination:\n\tKernel\tCost\tGamma\tCoef0\tDegree\tTotAcc.\tC1Acc.\tC2Acc."
                          << std::endl;

                char out_kernel;

                out_kernel = decide_kernel(row_index, linear_rows, radial_rows, sigmoid_rows, polynomial_rows);
                std::cout << "\t" << get_extended_kernel_name(out_kernel) << " ";

                print_vector(best_row, tuning_table_columns, false);
                // todo: save data somewhere

                free(best_row);
                free(accuracies);

            }

            if(tuning_logic) {
                free(local_tuning_table);
                local_tuning_table = nullptr;
            }
            free(final_tuning_table);
            final_tuning_table = nullptr;

            break;

        }

        case train_flag::undefined:
#if SHOW_LOGTIME
            logtime();
#endif
            std::cout << "[ERROR] Program logic was not passed!";
            exit(1);
    }


    if (performance_checks) {

        control = MPI_Barrier(MPI_COMM_WORLD);
        if (control != MPI_SUCCESS) { exit(1); }

        *(time_checks + time_iterator) = MPI_Wtime();


        if (process_rank == MASTER_PROCESS) {

            std::cout << std::endl;

#if SHOW_LOGTIME
            logtime();
#endif
            std::cout << "[INFO] Program ends at time " << *(time_checks + time_iterator) << std::endl;
#if SHOW_LOGTIME
            logtime();
#endif
            std::cout << "[INFO] Last step took " << *(time_checks + time_iterator) - *(time_checks + time_iterator - 1)
                      << " seconds\n" << std::endl;
        }
        // ++time_iterator; // end
    }

    free(time_checks);


    MPI_Finalize();

    return 0;

}