//
// Created by dmmp on 17/11/22.
//
#include "Dataset.h"
#include "svm_utils.h"
#include "mpi.h"

#ifndef HPC2022_TUNE_SVM_H
#define HPC2022_TUNE_SVM_H

void tune_linear(Dataset* df,
                 size_t training_break,
                 double* cost_array,
                 size_t cost_array_size,
                 double* result_table, /*output*/
                 size_t offset,
                 size_t result_table_columns,
                 int process_offset,
                 int available_processes)
{

}

void tune_radial(Dataset* df,
                 size_t training_break,
                 double* cost_array,
                 size_t cost_array_size,
                 double* gamma_array,
                 size_t gamma_array_size,
                 double* result_table, /*output*/
                 size_t offset,
                 size_t result_table_columns,
                 int process_offset,
                 int available_processes)
{

}

void tune_sigmoid(Dataset* df,
                 size_t training_break,
                 double* cost_array,
                 size_t cost_array_size,
                 double* gamma_array,
                 size_t gamma_array_size,
                 double* coef0_array,
                 size_t coef0_array_size,
                 double* result_table, /*output*/
                 size_t offset,
                 size_t result_table_columns,
                  int process_offset,
                  int available_processes)
{

}

void tune_polynomial(Dataset* df,
                 size_t training_break,
                 double* cost_array,
                 size_t cost_array_size,
                 double* gamma_array,
                 size_t gamma_array_size,
                 double* coef0_array,
                 size_t coef0_array_size,
                 double* degree_array,
                 size_t degree_array_size,
                 double* result_table, /*output*/
                 size_t offset,
                 size_t result_table_columns,
                     int process_offset,
                     int available_processes)
{

}




#endif //HPC2022_TUNE_SVM_H
