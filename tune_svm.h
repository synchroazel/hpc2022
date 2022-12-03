//
// Created by dmmp on 17/11/22.
//

#include "Dataset.h"
#include "svm_utils.h"
#include "mpi.h"

#ifndef HPC2022_TUNE_SVM_H
#define HPC2022_TUNE_SVM_H

#define DEBUG_TUNE false

double DEFAULT_COST_ARRAY[] = {0.001, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 100};
#define DEFAULT_COST_SIZE 10
double DEFAULT_GAMMA_ARRAY[] ={0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10};
#define DEFAULT_GAMMA_SIZE 8
double DEFAULT_COEF0_ARRAY[] ={0, 0.5, 1, 2.5, 5, 10};
#define DEFAULT_COEF0_SIZE 6
double DEFAULT_DEGREE_ARRAY[] ={1, 2, 3, 4, 5, 10};
#define DEFAULT_DEGREE_SIZE 6

char decide_kernel(int row_index, int linear_rows, int radial_rows, int sigmoid_rows, int polynomial_rows){
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
    return out_kernel;
}

void tune_linear(Dataset *df_train,
                 Dataset *df_validation,
                 const double *cost_array,
                 int cost_array_size,
                 double *result_table, /*output*/
                 int offset,
                 int result_table_columns,
                 int process_offset,
                 int available_processes,
                 double lr = DEFAULT_LEARNING_RATE,
                 double limit = DEFAULT_LIMIT,
                 double eps = DEFAULT_EPS,
                 bool verbose = false) {

    int current_process; // es: 21, with offset 20
    MPI_Comm_rank(MPI_COMM_WORLD, &current_process);

    int max_elements = cost_array_size; // es: 3000
    int chunk_size = (int)(ceil((double )(max_elements)/(double )(available_processes))); // es: 3000 / 8 = 375

    int start = (current_process - process_offset) * chunk_size; // es: (21 - 20) * 375 = 375
    int end = start + chunk_size; // es: 375 + 375 = 750
    if(end > max_elements){ end = max_elements;}

    int counter = 0;
    auto* combination_matrix = (double *) calloc(chunk_size * 1, sizeof(double )); // matrix

    // create table
    for(int i=0; i<cost_array_size && counter < end;i++){
        if(counter >= start)
        {
            combination_matrix[index(counter - start, 0, 1)] = cost_array[i];
        }
            ++counter;
    }


    for(int i = start; i<end; i++){
        Kernel_SVM svm;
        set_kernel_function(&svm, 'l');
        svm.verbose = verbose;

        double gamma = 0;
        double coef0 = 0;
        double degree = 0;
        double params[4] = {combination_matrix[index(i-start,0,2)],gamma, coef0, degree};

#if DEBUG_TUNE
        std::cout << "Process " << current_process << " training (radial) with cost " << params[0] << " and gamma " << params[1] << std::endl;
#endif
        serial_train(*df_train, &svm, params, lr, limit,false,"",0,eps);

        serial_test(*df_validation, &svm);

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



void tune_radial(Dataset *df_train,
                 Dataset *df_validation,
                 const double *cost_array,
                 int cost_array_size,
                 const double *gamma_array,
                 int gamma_array_size,
                 double *result_table, /*output*/
                 int offset,
                 int result_table_columns,
                 int process_offset,
                 int available_processes,

                 double lr=DEFAULT_LEARNING_RATE,
                 double limit= DEFAULT_LIMIT,
                 double eps= DEFAULT_EPS,
                 bool verbose= false)
{

#if DEBUG_TUNE



#endif

    int current_process; // es: 21, with offset 20
    MPI_Comm_rank(MPI_COMM_WORLD, &current_process);

    int max_elements = int(cost_array_size * gamma_array_size); // es: 3000
    int chunk_size = (int)(ceil((double )(max_elements)/(double )(available_processes))); // es: 3000 / 8 = 375

    int start = (current_process - process_offset) * chunk_size; // es: (21 - 20) * 375 = 375
    int end = start + chunk_size; // es: 375 + 375 = 750
    if(end > max_elements){ end = max_elements;}

    int counter = 0;
    auto* combination_matrix = (double *) calloc(chunk_size * 2, sizeof(double )); // matrix

    // create table
    for(int i=0; i<cost_array_size && counter < end;i++){
        for(int j=0;j<gamma_array_size && counter < end;j++){
            if(counter >= start)
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
        std::cout << "Process " << current_process << " training (radial) with cost " << params[0] << " and gamma " << params[1] << std::endl;
#endif
        serial_train(*df_train, &svm, params, lr, limit,false,"",0,eps);

        serial_test(*df_validation, &svm);

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
                  const double *cost_array,
                  int cost_array_size,
                  const double *gamma_array,
                  int gamma_array_size,
                  const double *coef0_array,
                  int coef0_array_size,
                  double *result_table, /*output*/
                  int offset,
                  int result_table_columns,
                  int process_offset,
                  int available_processes,

                  double lr = DEFAULT_LEARNING_RATE,
                  double limit = DEFAULT_LIMIT,
                  double eps = DEFAULT_EPS,
                  bool verbose= false) {

    int current_process; // es: 21, with offset 20
    MPI_Comm_rank(MPI_COMM_WORLD, &current_process);

    int max_elements = int(cost_array_size * gamma_array_size * coef0_array_size); // es: 3000
    int chunk_size = (int)(ceil((double )(max_elements)/(double )(available_processes))); // es: 3000 / 8 = 375

    int start = (current_process - process_offset) * chunk_size; // es: (21 - 20) * 375 = 375
    int end = start + chunk_size; // es: 375 + 375 = 750
    if(end > max_elements){ end = max_elements;}

    int counter = 0;
    auto* combination_matrix = (double *) calloc(chunk_size * 3, sizeof(double )); // matrix

    // create table
    for(int i=0; i<cost_array_size && counter < end;i++){
        for(int j=0;j<gamma_array_size && counter < end;j++){
            for(int k=0; k<coef0_array_size && counter < end; k++) {
                if (counter >= start) {
                    combination_matrix[index(counter - start, 0, 3)] = cost_array[i];
                    combination_matrix[index(counter - start, 1, 3)] = gamma_array[j];
                    combination_matrix[index(counter - start, 2, 3)] = coef0_array[k];
                }
                ++counter;
            }

        }
    }


    for(int i = start; i<end; i++){
        Kernel_SVM svm;
        set_kernel_function(&svm, 's');
        svm.verbose = verbose;
        double degree = 0;
        double params[4] = {combination_matrix[index(i-start,0,3)],
                            combination_matrix[index(i-start,1,3)],
                            combination_matrix[index(i-start,2,3)],
                            degree};

#if DEBUG_TUNE
        std::cout << "Process " << current_process << " training (sigmoid) with cost " << params[0] << " , gamma " << params[1] << " and intercept " << params[2] << std::endl;
#endif
        serial_train(*df_train, &svm, params, lr, limit,false,"",0,eps);

        serial_test(*df_validation, &svm);

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



void tune_polynomial(Dataset *df_train,
                     Dataset *df_validation,
                     const double *cost_array,
                     int cost_array_size,
                     const double *gamma_array,
                     int gamma_array_size,
                     const double *coef0_array,
                     int coef0_array_size,
                     const double *degree_array,
                     int degree_array_size,
                     double *result_table, /*output*/
                     int offset,
                     int result_table_columns,
                     int process_offset,
                     int available_processes,

                     double lr = DEFAULT_LEARNING_RATE,
                     double limit = DEFAULT_LIMIT,
                     double eps = DEFAULT_EPS,
                     bool verbose=false) {

    int current_process; // es: 21, with offset 20
    MPI_Comm_rank(MPI_COMM_WORLD, &current_process);

    int max_elements = int(cost_array_size * gamma_array_size * coef0_array_size * degree_array_size); // es: 3000
    int chunk_size = (int)(ceil((double )(max_elements)/(double )(available_processes))); // es: 3000 / 8 = 375

    int start = (current_process - process_offset) * chunk_size; // es: (21 - 20) * 375 = 375
    int end = start + chunk_size; // es: 375 + 375 = 750
    if(end > max_elements){ end = max_elements;}

    int counter = 0;
    auto* combination_matrix = (double *) calloc(chunk_size * 4, sizeof(double )); // matrix

    // create table
    for(int i=0; i<cost_array_size && counter < end;i++){
        for(int j=0;j<gamma_array_size && counter < end;j++){
            for(int k=0; k<coef0_array_size && counter < end; k++) {
                for(int l=0; l < degree_array_size && counter < end; l++ ) {
                    if (counter >= start) {
                        combination_matrix[index(counter - start, 0, 4)] = cost_array[i];
                        combination_matrix[index(counter - start, 1, 4)] = gamma_array[j];
                        combination_matrix[index(counter - start, 2, 4)] = coef0_array[k];
                        combination_matrix[index(counter - start, 3, 4)] = degree_array[l];
                    }
                    ++counter;
                }
            }

        }
    }


    for(int i = start; i<end; i++){
        Kernel_SVM svm;
        set_kernel_function(&svm, 'p');
        svm.verbose = verbose;
        double params[4] = {combination_matrix[index(i-start,0,4)],
                            combination_matrix[index(i-start,1,4)],
                            combination_matrix[index(i-start,2,4)],
                            combination_matrix[index(i-start,3,4)]};

#if DEBUG_TUNE
        std::cout << "Process " << current_process << " training (polynomial) with cost " << params[0] << " , gamma " << params[1] << ", intercept " << params[2]  << " and degree " << params[3] << std::endl;
#endif
        serial_train(*df_train, &svm, params, lr, limit,false,"",0,eps);

        serial_test(*df_validation, &svm);

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


void tune_linear2(Dataset *df_train,
                 Dataset *df_validation,
                 const double *cost_array,
                 int cost_array_size,
                 double *result_table, /*output*/
                 int offset,
                 int result_table_columns,
                 int process_offset,
                 int available_processes,
                 double lr = DEFAULT_LEARNING_RATE,
                 double limit = DEFAULT_LIMIT,
                 double eps = DEFAULT_EPS,
                 bool verbose = false) {

    int current_process; // es: 21, with offset 20
    MPI_Comm_rank(MPI_COMM_WORLD, &current_process);

    for(int i = 0; i<cost_array_size; i++){
        Kernel_SVM svm;
        set_kernel_function(&svm, 'l');
        svm.verbose = verbose;

        double gamma = 0;
        double coef0 = 0;
        double degree = 0;
        double params[4] = {cost_array[i],gamma, coef0, degree};

#if DEBUG_TUNE
        std::cout << "Process " << current_process << " training (radial) with cost " << params[0] << " and gamma " << params[1] << std::endl;
#endif
        parallel_train(*df_train, &svm, params, lr, limit,process_offset,available_processes,false,"",0,eps);

        // TODO: change to parallel
        parallel_test(*df_validation, &svm, process_offset, available_processes);


        result_table[index(offset + i, 0, result_table_columns)] = params[0];
        result_table[index(offset + i , 1, result_table_columns)] = params[1];
        result_table[index(offset + i , 2, result_table_columns)] = params[2];
        result_table[index(offset + i , 3, result_table_columns)] = params[3];

        result_table[index(offset + i , 4, result_table_columns)] = svm.accuracy;
        result_table[index(offset + i , 5, result_table_columns)] = svm.accuracy_c1;
        result_table[index(offset + i , 6, result_table_columns)] = svm.accuracy_c2;
    }
}



void tune_radial2(Dataset *df_train,
                 Dataset *df_validation,
                 const double *cost_array,
                 int cost_array_size,
                 const double *gamma_array,
                 int gamma_array_size,
                 double *result_table, /*output*/
                 int offset,
                 int result_table_columns,
                 int process_offset,
                 int available_processes,

                 double lr=DEFAULT_LEARNING_RATE,
                 double limit= DEFAULT_LIMIT,
                 double eps= DEFAULT_EPS,
                 bool verbose= false)
{


    int current_process; // es: 21, with offset 20
    MPI_Comm_rank(MPI_COMM_WORLD, &current_process);



    for(int i = 0; i<cost_array_size; i++) {
        for (int j = 0; j < gamma_array_size; j++) {
            Kernel_SVM svm;
            set_kernel_function(&svm, 'r');
            svm.verbose = verbose;

            double coef0 = 0;
            double degree = 0;
            double params[4] = {cost_array[i], gamma_array[j],
                                coef0, degree};

#if DEBUG_TUNE
            std::cout << "Process " << current_process << " training (radial) with cost " << params[0] << " and gamma " << params[1] << std::endl;
#endif
            parallel_train(*df_train, &svm, params, lr, limit,process_offset,available_processes,false,"",0,eps);

            // TODO: change to parallel
            parallel_test(*df_validation, &svm, process_offset, available_processes);

            result_table[index(offset + i, 0, result_table_columns)] = params[0];
            result_table[index(offset + i, 1, result_table_columns)] = params[1];
            result_table[index(offset + i, 2, result_table_columns)] = params[2];
            result_table[index(offset + i, 3, result_table_columns)] = params[3];

            result_table[index(offset + i, 4, result_table_columns)] = svm.accuracy;
            result_table[index(offset + i, 5, result_table_columns)] = svm.accuracy_c1;
            result_table[index(offset + i, 6, result_table_columns)] = svm.accuracy_c2;
        }
    }

}


void tune_sigmoid2(Dataset *df_train,
                  Dataset *df_validation,
                  const double *cost_array,
                  int cost_array_size,
                  const double *gamma_array,
                  int gamma_array_size,
                  const double *coef0_array,
                  int coef0_array_size,
                  double *result_table, /*output*/
                  int offset,
                  int result_table_columns,
                  int process_offset,
                  int available_processes,

                  double lr = DEFAULT_LEARNING_RATE,
                  double limit = DEFAULT_LIMIT,
                  double eps = DEFAULT_EPS,
                  bool verbose= false) {

    int current_process; // es: 21, with offset 20
    MPI_Comm_rank(MPI_COMM_WORLD, &current_process);




    for(int i = 0; i<cost_array_size; i++) {
        for (int j = 0; j < gamma_array_size; j++) {
            for (int k = 0; k < coef0_array_size; k++) {

                Kernel_SVM svm;
                set_kernel_function(&svm, 's');
                svm.verbose = verbose;
                double degree = 0;
                double params[4] = {cost_array[i],
                                    gamma_array[j],
                                    coef0_array[k],
                                    degree};

#if DEBUG_TUNE
                std::cout << "Process " << current_process << " training (sigmoid) with cost " << params[0] << " , gamma " << params[1] << " and intercept " << params[2] << std::endl;
#endif
                parallel_train(*df_train, &svm, params, lr, limit,process_offset,available_processes,false,"",0,eps);

                // TODO: change to parallel
                parallel_test(*df_validation, &svm, process_offset, available_processes);

                result_table[index(offset + i * gamma_array_size * coef0_array_size + j * coef0_array_size + k, 0, result_table_columns)] = params[0];
                result_table[index(offset + i * gamma_array_size * coef0_array_size + j * coef0_array_size + k, 1, result_table_columns)] = params[1];
                result_table[index(offset + i * gamma_array_size * coef0_array_size + j * coef0_array_size + k, 2, result_table_columns)] = params[2];
                result_table[index(offset + i * gamma_array_size * coef0_array_size + j * coef0_array_size + k, 3, result_table_columns)] = params[3];

                result_table[index(offset + i * gamma_array_size * coef0_array_size + j * coef0_array_size + k, 4, result_table_columns)] = svm.accuracy;
                result_table[index(offset + i * gamma_array_size * coef0_array_size + j * coef0_array_size + k, 5, result_table_columns)] = svm.accuracy_c1;
                result_table[index(offset + i * gamma_array_size * coef0_array_size + j * coef0_array_size + k, 6, result_table_columns)] = svm.accuracy_c2;

            }

        }
    }
}



void tune_polynomial2(Dataset *df_train,
                     Dataset *df_validation,
                     const double *cost_array,
                     int cost_array_size,
                     const double *gamma_array,
                     int gamma_array_size,
                     const double *coef0_array,
                     int coef0_array_size,
                     const double *degree_array,
                     int degree_array_size,
                     double *result_table, /*output*/
                     int offset,
                     int result_table_columns,
                     int process_offset,
                     int available_processes,

                     double lr = DEFAULT_LEARNING_RATE,
                     double limit = DEFAULT_LIMIT,
                     double eps = DEFAULT_EPS,
                     bool verbose=false) {

    int current_process; // es: 21, with offset 20
    MPI_Comm_rank(MPI_COMM_WORLD, &current_process);



    for(int i = 0; i<cost_array_size; i++) {
        for (int j = 0; j < gamma_array_size; j++) {
            for (int k = 0; k < coef0_array_size; k++) {
                for (int l = 0; l < degree_array_size; l++) {


                    Kernel_SVM svm;
                    set_kernel_function(&svm, 'p');
                    svm.verbose = verbose;
                    double params[4] = {cost_array[i],
                                        gamma_array[j],
                                        coef0_array[k],
                                        degree_array[l]};

#if DEBUG_TUNE
                    std::cout << "Process " << current_process << " training (polynomial) with cost " << params[0] << " , gamma " << params[1] << ", intercept " << params[2]  << " and degree " << params[3] << std::endl;
#endif
                    parallel_train(*df_train, &svm, params, lr, limit,process_offset,available_processes,false,"",0,eps);

                    // TODO: change to parallel
                    parallel_test(*df_validation, &svm, process_offset, available_processes);

                    result_table[index(offset + i, 0, result_table_columns)] = params[0];
                    result_table[index(offset + i, 1, result_table_columns)] = params[1];
                    result_table[index(offset + i, 2, result_table_columns)] = params[2];
                    result_table[index(offset + i, 3, result_table_columns)] = params[3];

                    result_table[index(offset + i, 4, result_table_columns)] = svm.accuracy;
                    result_table[index(offset + i, 5, result_table_columns)] = svm.accuracy_c1;
                    result_table[index(offset + i, 6, result_table_columns)] = svm.accuracy_c2;
                }

            }
        }
    }

}


#endif //HPC2022_TUNE_SVM_H
