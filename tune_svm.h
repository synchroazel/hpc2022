//
// Created by dmmp on 17/11/22.
//

#include "Dataset.h"
#include "svm_utils.h"
#include "mpi.h"

#ifndef HPC2022_TUNE_SVM_H
#define HPC2022_TUNE_SVM_H

#define DEBUG_TUNE true

void tune_linear(Dataset *df_train,
                 Dataset *df_validation,
                 double *cost_array,
                 size_t cost_array_size,
                 double *result_table, /*output*/
                 size_t offset,
                 size_t result_table_columns,
                 int process_offset,
                 int available_processes,
                 double lr = DEFAULT_LEARNING_RATE,
                 double limit = DEFAULT_LIMIT,
                 double eps = DEFAULT_EPS,
                 bool verbose = false) {


    int cost_size_per_process = (int)std::ceil((double )cost_array_size / (double )available_processes);
    // es 1:                                                  8         /            16                =   1
    // es 2:                                                 16        /            7               =   2

    int current_process; // es: 21
    MPI_Comm_rank(MPI_COMM_WORLD, &current_process);

#if DEBUG_TUNE
    std::cout << "I'm process " << current_process << " and I will operate from " << (current_process - process_offset) * cost_size_per_process << " to " << (((current_process - process_offset) * cost_size_per_process) + cost_size_per_process) - 1 << std::endl;
#endif


    //es:        (21           -      20)        *          3
    for (int i = (current_process - process_offset) * cost_size_per_process;
        //es:           ((21         -    20)          *          3)=3         +          3   =6
         i < (((current_process - process_offset) * cost_size_per_process) + cost_size_per_process) &&
         i < cost_array_size;
         i++) {

        Kernel_SVM svm;
        set_kernel_function(&svm, 'l');
        svm.verbose = verbose;

        double gamma = 0;
        double coef0 = 0;
        double degree = 0;
        double params[4] = {cost_array[i], gamma, coef0, degree};
#if DEBUG_TUNE
            std::cout << "Process " << current_process << " training cost " << params[0] << std::endl;
#endif

            train(*df_train, &svm, params, lr, limit,current_process,1,false,"",0,eps);


            test(*df_validation, &svm, current_process, 1);

            result_table[index(offset + i, 0, result_table_columns)] = params[0];
            result_table[index(offset + i, 1, result_table_columns)] = params[1];
            result_table[index(offset + i, 2, result_table_columns)] = params[2];
            result_table[index(offset + i, 3, result_table_columns)] = params[3];

            result_table[index(offset + i, 4, result_table_columns)] = svm.accuracy;
            result_table[index(offset + i, 5, result_table_columns)] = svm.accuracy_c1;
            result_table[index(offset + i, 6, result_table_columns)] = svm.accuracy_c2;

#if DEBUG_TUNE
            double tmp[result_table_columns];
        get_row(result_table, i, result_table_columns, tmp);
        print_vector(tmp,result_table_columns)  ;
#endif

        }
}



void tune_radial(Dataset *df_train,
                 Dataset *df_validation,
                 double *cost_array,
                 size_t cost_array_size,
                 double *gamma_array,
                 size_t gamma_array_size,
                 double *result_table, /*output*/
                 size_t offset,
                 size_t result_table_columns,
                 int process_offset,
                 int available_processes,

                 double lr=DEFAULT_LEARNING_RATE,
                 double limit= DEFAULT_LIMIT,
                 double eps= DEFAULT_EPS,
                 bool verbose= false)
{

    int current_process; // es: 21
    MPI_Comm_rank(MPI_COMM_WORLD, &current_process);

    bool too_many_processes = cost_array_size * gamma_array_size < available_processes;

    if(too_many_processes){
#if DEBUG_TUNE
        std::cout << "There are too many processes. Logic 1: each process will cover 1 training " << std::endl;
#endif
        //-----------------------------------------------------------------------------------------------------------
        int iteration = (int)(current_process - (int)process_offset / cost_array_size);
        if(iteration >= gamma_array_size){
            return;
        }
        int i = (int)(current_process - process_offset) % (int)cost_array_size;
        int j = i + int(cost_array_size * iteration);

        Kernel_SVM svm;
        set_kernel_function(&svm, 'r');
        svm.verbose = verbose;

        double coef0 = 0;
        double degree = 0;
        double params[4] = {cost_array[i], gamma_array[j], coef0, degree};

#if DEBUG_TUNE
        std::cout << "Process " << current_process << " training with cost " << params[0] << " and gamma " << params[1] << std::endl;
#endif
        train(*df_train, &svm, params, lr, limit,current_process,1,false,"",0,eps);


        test(*df_validation, &svm, current_process,1);

        // TODO: correggere
        result_table[index(offset + i * gamma_array_size + j, 0, result_table_columns)] = params[0];
        result_table[index(offset + i * gamma_array_size + j, 1, result_table_columns)] = params[1];
        result_table[index(offset + i * gamma_array_size + j, 2, result_table_columns)] = params[2];
        result_table[index(offset + i * gamma_array_size + j, 3, result_table_columns)] = params[3];

        result_table[index(offset + i * gamma_array_size + j, 4, result_table_columns)] = svm.accuracy;
        result_table[index(offset + i * gamma_array_size + j, 5, result_table_columns)] = svm.accuracy_c1;
        result_table[index(offset + i * gamma_array_size + j, 6, result_table_columns)] = svm.accuracy_c2;

        //---------------------------------------------------------------------------------------------------------
    } else if(available_processes < cost_array_size){ // too few processes
#if DEBUG_TUNE
        std::cout << "There are too few processes. Logic 2: each process will split the cost and do all gammas " << std::endl;
        std::cout << "Available processes: " << available_processes << ", cost array size: " << cost_array_size << std::endl;
#endif

        int cost_size_per_process = (int)std::ceil((double )cost_array_size / (double )available_processes);
        for (int i = (current_process - process_offset) * cost_size_per_process;
            //es:           ((21         -    20)          *          3)=3         +          3   =6
             i < (((current_process - process_offset) * cost_size_per_process) + cost_size_per_process) &&
             i < cost_array_size;
             i++) {
            // es1 : process 20, with offset 19 and 15 available. Cost size = 8
            //  i=      (20-19)1  % 8 = 1
            // processes

            // es1 : process 20, with offset 10 and 15 available. Cost size = 8
            //  i=      (20-10)10  % 8 = 2

            for(int j=0; j<gamma_array_size; j++){

                Kernel_SVM svm;
                set_kernel_function(&svm, 'r');
                svm.verbose = verbose;

                double coef0 = 0;
                double degree = 0;
                double params[4] = {cost_array[i], gamma_array[j], coef0, degree};

#if DEBUG_TUNE
                std::cout << "Process " << current_process << ", training cost " << params[0] << " Gamma " << params[1] << std::endl;
#endif
                train(*df_train, &svm, params, lr, limit,current_process,1,false,"",0,eps);


                test(*df_validation, &svm, current_process,1);

                result_table[index(offset + i * gamma_array_size + j, 0, result_table_columns)] = params[0];
                result_table[index(offset + i * gamma_array_size + j, 1, result_table_columns)] = params[1];
                result_table[index(offset + i * gamma_array_size + j, 2, result_table_columns)] = params[2];
                result_table[index(offset + i * gamma_array_size + j, 3, result_table_columns)] = params[3];

                result_table[index(offset + i * gamma_array_size + j, 4, result_table_columns)] = svm.accuracy;
                result_table[index(offset + i * gamma_array_size + j, 5, result_table_columns)] = svm.accuracy_c1;
                result_table[index(offset + i * gamma_array_size + j, 6, result_table_columns)] = svm.accuracy_c2;
            }
        }
        //--------------------------------------------------------------------------------------------------------------
    } else {

        int max_iters = (int)(available_processes / cost_array_size);
        int i_trigger = (int)(available_processes % cost_array_size);
        int current_iter = (int)((current_process - process_offset) / cost_array_size);
        int j_blocks = (int)gamma_array_size / (max_iters+1);

#if DEBUG_TUNE


        if(current_process == MASTER_PROCESS){
            std::cout << "Most common case, logic 3. Currently, there are " << available_processes << " available processes." << std::endl;
            std::cout << "Max iterations:  " << max_iters << std::endl << "i trigger:  " << i_trigger << std::endl << "j blocks:  " << j_blocks << std::endl;

        }

        std::cout << "Process: " << current_process<< ", current iteration:  " << current_iter << std::endl;

#endif

        for(int i = (int)(current_process - process_offset) % (int)cost_array_size; i< ((int)(current_process - process_offset) % (int)cost_array_size)+1; i++){
#if DEBUG_TUNE
            std::cout << "I'm process " << current_process << " and will cover cost" << cost_array[i] << std::endl;
#endif
            //-----------------------------------------------------------------------------------------------
            if((i >= i_trigger) && (current_iter < max_iters) ){

#if DEBUG_TUNE
                std::cout << "Current i: " << i << ", current iter= " <<  current_iter << std::endl << "Trigger has been reached! " << std::endl << "I'm process " << current_process << " and will cover all j" << std::endl;

#endif

                for(int j=0; j<gamma_array_size; j++){
                    Kernel_SVM svm;
                    set_kernel_function(&svm, 'r');
                    svm.verbose = verbose;

                    double coef0 = 0;
                    double degree = 0;
                    double params[4] = {cost_array[i], gamma_array[j], coef0, degree};
                    train(*df_train, &svm, params, lr, limit,current_process,1,false,"",0,eps);


                    test(*df_validation, &svm, current_process,1);

#if DEBUG_TUNE
                    std::cout << "\nProcess " << current_process << " writing to row " << offset + i * cost_array_size + j << "\n" << "Offset: " << offset<< " i: " << i << " j: " << j << "\n" << std::endl;

#endif

                    result_table[index(offset + i * gamma_array_size + j, 0, result_table_columns)] = params[0];
                    result_table[index(offset + i * gamma_array_size + j, 1, result_table_columns)] = params[1];
                    result_table[index(offset + i * gamma_array_size + j, 2, result_table_columns)] = params[2];
                    result_table[index(offset + i * gamma_array_size + j, 3, result_table_columns)] = params[3];

                    result_table[index(offset + i * gamma_array_size + j, 4, result_table_columns)] = svm.accuracy;
                    result_table[index(offset + i * gamma_array_size + j, 5, result_table_columns)] = svm.accuracy_c1;
                    result_table[index(offset + i * gamma_array_size + j, 6, result_table_columns)] = svm.accuracy_c2;
                }
            //-----------------------------------------------------------------------------------------------
            } else {
#if DEBUG_TUNE
                std::cout << "Trigger has not been reached " << std::endl;
                std::cout << "I'm process " << current_process << " and will cover j sections from " << (int)((current_process - process_offset) / ( cost_array_size)) * (current_iter * j_blocks) << " to " <<  j_blocks * (current_iter + 1) -1 << std::endl;
#endif

                for(int j= (int)((current_process - process_offset) / ( cost_array_size)) * (current_iter * j_blocks);
                (j < j_blocks * (current_iter + 1)) && (j < gamma_array_size);
                j++){


                    Kernel_SVM svm;
                    set_kernel_function(&svm, 'r');
                    svm.verbose = verbose;

                    double coef0 = 0;
                    double degree = 0;
                    double params[4] = {cost_array[i], gamma_array[j], coef0, degree};
                    train(*df_train, &svm, params, lr, limit,current_process,1,false,"",0,eps);


                    test(*df_validation, &svm, current_process,1);

#if DEBUG_TUNE
                    std::cout << "\nProcess " << current_process << " writing to row " << offset + i * cost_array_size + j << "\n" << "Offset: " << offset<< " i: " << i << " j: " << j << "\n" << std::endl;

#endif


                    result_table[index(offset + i * gamma_array_size+ j, 0, result_table_columns)] = params[0];
                    result_table[index(offset + i * gamma_array_size + j, 1, result_table_columns)] = params[1];
                    result_table[index(offset + i * gamma_array_size + j, 2, result_table_columns)] = params[2];
                    result_table[index(offset + i * gamma_array_size + j, 3, result_table_columns)] = params[3];

                    result_table[index(offset + i * gamma_array_size + j, 4, result_table_columns)] = svm.accuracy;
                    result_table[index(offset + i * gamma_array_size + j, 5, result_table_columns)] = svm.accuracy_c1;
                    result_table[index(offset + i * gamma_array_size + j, 6, result_table_columns)] = svm.accuracy_c2;
                }
            }



        }
    }

}


void tune_sigmoid(Dataset *df_train,
                  Dataset *df_validation,
                  double *cost_array,
                  size_t cost_array_size,
                  double *gamma_array,
                  size_t gamma_array_size,
                  double *coef0_array,
                  size_t coef0_array_size,
                  double *result_table, /*output*/
                  size_t offset,
                  size_t result_table_columns,
                  int process_offset,
                  int available_processes,
                  double lr = DEFAULT_LEARNING_RATE,
                  double limit = DEFAULT_LIMIT,
                  double eps = DEFAULT_EPS) {

}

void tune_polynomial(Dataset *df_train,
                     Dataset *df_validation,
                     double *cost_array,
                     size_t cost_array_size,
                     double *gamma_array,
                     size_t gamma_array_size,
                     double *coef0_array,
                     size_t coef0_array_size,
                     double *degree_array,
                     size_t degree_array_size,
                     double *result_table, /*output*/
                     size_t offset,
                     size_t result_table_columns,
                     int process_offset,
                     int available_processes,
                     double lr = DEFAULT_LEARNING_RATE,
                     double limit = DEFAULT_LIMIT,
                     double eps = DEFAULT_EPS) {

}

#endif //HPC2022_TUNE_SVM_H
