#include "iostream"
#include "mpi.h"

#include "Dataset.h"

#include "tune_svm.h"

#include "read_dataset.h"


/* Macro to switch modes */
#define DEBUG_TRAIN false
#define DEBUG_TEST false
#define CLI_ARGS false
#define IMPLEMENTED_KERNELS 4
#define NUMBER_OF_HYPER_PARAMETERS 4

//todo:
//      implement:
//                performance check
//                cli args
//                open_mp
//                validation in tuning functions
//                linear tuning
//                radial tuning
//                sigmoid tuning
//                poly tuning
//                all together logic
//      debug:
//                read and write from binary file
//                svm test
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

    enum relevant_metric {Accuracy = NUMBER_OF_HYPER_PARAMETERS + 1, AccuracyC1 = NUMBER_OF_HYPER_PARAMETERS +2, AccuracyC3 = NUMBER_OF_HYPER_PARAMETERS +3};


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
    std::string filepath = "/home/dmmp/Documents/GitHub/hpc2022/data/iris_train.csv";
    // std::string filepath = "../data/iris_test.csv";
    size_t rows = 70, columns = 5, target_column = 5;
    double train_percentage = 0.8;

    if(process_rank == MASTER_PROCESS){
        std::cout << "Dataset filepath: " << filepath << std::endl;
        std::cout << "The dataset has " << rows << " rows and " << columns << " columns." << std::endl;
    }


    size_t cost_array_size = 10, gamma_array_size = 8, coef0_array_size = 6, degree_array_size=7;
    double cost_array[] = {0.001, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 100};
    double gamma_array[] = { 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10};
    double coef0_array[] = {0,0.5,1,2.5,5,10};
    double degree_array[] = {1, 2, 3,4,5,10, static_cast<double>(columns-1)};

    bool train_flag;
#endif


#if DEBUG_TRAIN
    std::string filepath = "/home/dmmp/Documents/GitHub/hpc2022/data/iris_train.csv";

    Dataset df_train = read_dataset(filepath, 70, 5, 5);

    // TODO: change to parallel logic
    if (process_rank == MASTER_PROCESS) {

       char ker_type = 'p';

       Kernel_SVM svm;
       set_kernel_function(&svm, ker_type);
       svm.verbose= true;

       double Cost = 5;
       double gamma = 0.1;
       double coef0 = 0;
       double degree = 1;
       double params[4] = {Cost,gamma,coef0,degree};

       double lr = 0.0001;
       double limit = 0.1;

       train(df_train, &svm, params, lr, limit);
    }

#elif DEBUG_TEST
    // TODO
#else



    Dataset df = read_dataset(filepath, rows, columns, target_column);

    if(train_flag){
        // TODO: refactor?
        // code for training

        int train_break = (int) ceil((double)(rows) * train_percentage );

        size_t linear_rows = cost_array_size;
        size_t radial_rows = cost_array_size * gamma_array_size;
        size_t sigmoid_rows = cost_array_size * gamma_array_size * coef0_array_size;
        size_t polynomial_rows = cost_array_size * gamma_array_size * coef0_array_size * degree_array_size;

        size_t tuning_table_rows = linear_rows + radial_rows + sigmoid_rows + polynomial_rows;
        size_t tuning_table_columns = NUMBER_OF_HYPER_PARAMETERS + 1/* accuracy*/ + 1/*class 1 accuracy*/ + 1/*class 2 accuracy*/; // NB: type of kernel will be printed separately
        if(process_rank == MASTER_PROCESS){
            std::cout << "There are a total of " << IMPLEMENTED_KERNELS << " kernels to tune." << std::endl;
            std::cout << "Tuning will use:\n\t " << cost_array_size << " different costs, " << std::endl;
            std::cout << "\t " << gamma_array_size << " different gamma, " << std::endl;
            std::cout << "\t " << coef0_array_size << " different intercepts, " << std::endl;
            std::cout << "\t " << degree_array_size << " different exponential degrees, " << std::endl;
            std::cout << "For a total of " << tuning_table_rows << " combinations" << std::endl;
        }

        double final_tuning_table[tuning_table_rows * tuning_table_columns]; // matrix

        // TODO: decide condition
        if(world_size < 4){
            // one after the other

            //linear
            if(process_rank == MASTER_PROCESS){
                std::cout << "Starting linear tuning" << std::endl;
            }
#if PERFORMANCE_CHECK
            MPI_Barrier(MPI_COMM_WORLD);
#endif
            tune_linear(&df, train_break, cost_array, cost_array_size, final_tuning_table, 0, tuning_table_columns, MASTER_PROCESS, world_size);

            //radial
#if PERFORMANCE_CHECK
            MPI_Barrier(MPI_COMM_WORLD);
#endif
            if(process_rank == MASTER_PROCESS){
                std::cout << "Starting radial tuning" << std::endl;
            }
            tune_radial(&df, train_break, cost_array, cost_array_size, gamma_array, gamma_array_size,final_tuning_table, linear_rows * tuning_table_columns, tuning_table_columns, MASTER_PROCESS, world_size);

            //sigmoid
#if PERFORMANCE_CHECK
            MPI_Barrier(MPI_COMM_WORLD);
#endif
            if(process_rank == MASTER_PROCESS){
                std::cout << "Starting sigmoid tuning" << std::endl;
            }
            tune_sigmoid(&df, train_break, cost_array, cost_array_size, gamma_array, gamma_array_size, coef0_array, coef0_array_size, final_tuning_table, linear_rows * tuning_table_columns + radial_rows * tuning_table_columns, tuning_table_columns, MASTER_PROCESS, world_size);

            //polynomial
#if PERFORMANCE_CHECK
            MPI_Barrier(MPI_COMM_WORLD);
#endif
            if(process_rank == MASTER_PROCESS){
                std::cout << "Starting polynomial tuning" << std::endl;
            }
            tune_polynomial(&df, train_break, cost_array, cost_array_size, gamma_array, gamma_array_size, coef0_array, coef0_array_size,degree_array, degree_array_size, final_tuning_table, linear_rows * tuning_table_columns + radial_rows * tuning_table_columns +sigmoid_rows * tuning_table_columns , tuning_table_columns, MASTER_PROCESS, world_size);


        } else {
            // together, will require a gather
        }



    } else {
        // code for test
    }

    if (process_rank == MASTER_PROCESS) {

        /*std::string ker_type = "rbf";

        KernelFunc K;
        std::vector<double> params = {1};
        Set_Kernel(ker_type, K, params);

        // deserialize object from file savedata
        Kernel_SVM svm;


        {
            std::ifstream infile("../model.dat");
            boost::archive::text_iarchive archive(infile);
            archive >> svm;

        }


        // svm.test(df_test);


        std::cout << "\n\nExecution completed." << std::endl;

*/

//        /* CHECK IF ALL DATA WAS IMPORTED SUCCESSFULLY */
//
//        std::cout << "\n\narr_xs: " << std::endl;
//        for (int i = 0; i < 90; i++) {
//            for (int j = 0; j < 4; j++) {
//                std::cout << svm.arr_xs[i][j] << "   ";
//            }
//            std::cout << std::endl;
//        }
//
//        std::cout << "\n\narr_ys: " << std::endl;
//        for (int i = 0; i < 90; i++) {
//            std::cout << svm.arr_ys[i] << std::endl;
//        }
//
//        std::cout << "\n\narr_alpha_s: " << std::endl;
//        for (int i = 0; i < 90; i++) {
//            std::cout << svm.arr_alpha_s[i] << std::endl;
//        }
//
//        std::cout << "\n\narr_xs_in: " << std::endl;
//        for (int i = 0; i < 90; i++) {
//            for (int j = 0; j < 4; j++) {
//                std::cout << svm.arr_xs_in[i][j] << "   ";
//            }
//            std::cout << std::endl;
//        }
//
//        std::cout << "\n\narr_ys_in: " << std::endl;
//        for (int i = 0; i < 90; i++) {
//            std::cout << svm.arr_ys_in[i] << std::endl;
//        }
//
//        std::cout << "\n\narr_alpha_s_in: " << std::endl;
//        for (int i = 0; i < 90; i++) {
//            std::cout << svm.arr_alpha_s_in[i] << std::endl;
//        }
//
//        std::cout << "\n\nb: " << svm.b << std::endl;





    }


#endif

    MPI_Finalize();

    return 0;

}