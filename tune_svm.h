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
                 const double *cost_array,
                 size_t cost_array_size,
                 const double *gamma_array,
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

    int current_process; // es: 21, with offset 20
    MPI_Comm_rank(MPI_COMM_WORLD, &current_process);

    int max_elements = int(cost_array_size * gamma_array_size); // es: 3000
    int chunk_size = (int)(ceil((double )(max_elements)/(double )(available_processes))); // es: 3000 / 8 = 375

    int start = (current_process - process_offset) * chunk_size; // es: (21 - 20) * 375 = 375
    int end = start + chunk_size; // es: 375 + 375 = 750

    int counter = 0;
    auto* combination_matrix = (double *) calloc(chunk_size * 2, sizeof(double )); // matrix

    // create table
    for(int i=0; i<cost_array_size && counter < end;i++){
        for(int j=0;j<gamma_array_size && counter < end;j++){
            if(counter > start)
            {
                combination_matrix[index(counter - start, 0, 2)] = cost_array[i];
                combination_matrix[index(counter - start, 1, 2)] = gamma_array[j];
            }
            ++counter;

        }
    }


    for(int i = start; i<end; i++){
        Kernel_SVM svm;
        set_kernel_function(&svm, 'r');
        svm.verbose = verbose;

        double coef0 = 0;
        double degree = 0;
        double params[4] = {combination_matrix[index(i-start,0,2)],combination_matrix[index(i-start,1,2)], coef0, degree};

#if DEBUG_TUNE
        std::cout << "Process " << current_process << " training with cost " << params[0] << " and gamma " << params[1] << std::endl;
#endif
        train(*df_train, &svm, params, lr, limit,current_process,1,false,"",0,eps);


        test(*df_validation, &svm, current_process,1);

        // TODO: debug
        result_table[index(offset + i, 0, result_table_columns)] = params[0];
        result_table[index(offset + i , 1, result_table_columns)] = params[1];
        result_table[index(offset + i , 2, result_table_columns)] = params[2];
        result_table[index(offset + i , 3, result_table_columns)] = params[3];

        result_table[index(offset + i , 4, result_table_columns)] = svm.accuracy;
        result_table[index(offset + i , 5, result_table_columns)] = svm.accuracy_c1;
        result_table[index(offset + i , 6, result_table_columns)] = svm.accuracy_c2;
    }

    free(combination_matrix);
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
