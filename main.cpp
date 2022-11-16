#include "iostream"
#include "mpi.h"

#include "Dataset.h"

//#include "svm.hpp"
#include "svm.cpp"

#include "read_dataset.h"


/* Macro to switch modes */
#define DEBUG_TRAIN true
#define CLI_ARGS false


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
#endif


#if DEBUG_TRAIN
    std::string filepath = "/home/dmmp/Documents/GitHub/hpc2022/data/iris_train.csv";

    Dataset df_train = read_dataset(filepath, 70, 5, 5);

    // TODO: change to new logic
    if (process_rank == MASTER_PROCESS) {

       char ker_type = 'l';

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

#else

    std::string filepath = "/Users/azel/Developer/hpc2022/data/iris_test.csv";

    Dataset df_test = read_dataset(filepath, 30, 5, 5);

    if (process_rank == MASTER_PROCESS) {

        std::string ker_type = "rbf";

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