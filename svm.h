//
// Created by dmmp on 15/11/22.
//

#ifndef HPC2022_SVM_H
#define HPC2022_SVM_H

#include <string>
#include <cstdio>
#include "Dataset.h"


#define DEFAULT_LEARNING_RATE 0.001
#define DEFAULT_LIMIT 0.1
#define DEFAULT_EPS 1e-6


/**
 * Kernel namespace
 */

namespace kernel {

    double linear(const double *x1, const double *x2, int size, double params[4]); // MMC

    double polynomial(const double *x1, const double *x2, int size, double params[4]); // polynomial

    double radial(const double *x1, const double *x2, int size, const double params[4]); // radial basis

    double sigmoid(const double *x1, const double *x2, int size, const double params[4]); // radial basis
}

// define types for struct functions
typedef double (*KernelFunc)(double *, double *, int, double params[4]);


/**
 * Struct for SVM object
 */

typedef struct Kernel_SVM {

    // NB: _in = inside margin

    // stored for serialization
    int arr_xs_row_size{};
    int arr_xs_column_size{};
    int arr_xs_in_row_size{};

    double *arr_xs{};  // matrix
    int *arr_ys{};
    double *arr_alpha_s{};
    double *arr_xs_in{};  //matrix
    int *arr_ys_in{};
    double *arr_alpha_s_in{};
    double b{}; // bias

    double accuracy{};
    double accuracy_c1{}, accuracy_c2{};
    int correct_c1{}, correct_c2{};


    double params[4] = {1, 3, 0, 2};  // cost, gamma, coef0, degree
    bool verbose{};

    char kernel_type{};
    KernelFunc K{}; // linear, sigmoid, radial, polynomial
    //TrainFunc train{};//(Dataset training_data, double params[4], const double lr, const double limit = 0.001);

    //TestFunc test{};//(Dataset test_data);

    //HelperFunc f{};//(const double x); // used in train

    //HelperFunc g{};//(const double x); // used in test


} Kernel_SVM;

void set_kernel_function(Kernel_SVM *svm, char kernel_type) {
    svm->kernel_type = kernel_type;
    if (kernel_type == 'l') {
        svm->K = reinterpret_cast<KernelFunc>(kernel::linear);
    } else if (kernel_type == 'r') {
        svm->K = reinterpret_cast<KernelFunc>(kernel::radial);
    } else if (kernel_type == 's') {
        svm->K = reinterpret_cast<KernelFunc>(kernel::sigmoid);
    } else if (kernel_type == 'p') {
        svm->K = reinterpret_cast<KernelFunc>(kernel::polynomial);
    } else {
        std::cout << "That kernel function han not been implemented yet." << std::endl;
        exit(1);
    }
}
// TODO: add error checks


/**
 * Support for serialization
 */

int save_svm(const Kernel_SVM *svm, const std::string &path) {

    FILE *file_to_write;
    file_to_write = fopen(path.c_str(), "wb");

    if (!file_to_write) {
        std::cout << "Error opening file. Saving was not possible !";
        return 1;
    }

    // write sizes
    fwrite(&svm->arr_xs_row_size, sizeof(int), 1, file_to_write);
    fwrite(&svm->arr_xs_column_size, sizeof(int), 1, file_to_write);
    fwrite(&svm->arr_xs_in_row_size, sizeof(int), 1, file_to_write);

    // writing vectors
    fwrite(svm->arr_xs, sizeof(double) * svm->arr_xs_row_size * svm->arr_xs_column_size, 1, file_to_write);
    fwrite(svm->arr_ys, sizeof(int) * svm->arr_xs_row_size, 1, file_to_write);
    fwrite(svm->arr_alpha_s, sizeof(double) * svm->arr_xs_row_size, 1, file_to_write);
    fwrite(svm->arr_xs_in, sizeof(double) * svm->arr_xs_in_row_size * svm->arr_xs_column_size, 1, file_to_write);
    fwrite(svm->arr_ys_in, sizeof(int) * svm->arr_xs_in_row_size, 1, file_to_write);
    fwrite(svm->arr_alpha_s_in, sizeof(double) * svm->arr_xs_in_row_size, 1, file_to_write);

    // b
    fwrite(&svm->b, sizeof(double), 1, file_to_write);
    //accuracies
    fwrite(&svm->accuracy, sizeof(double), 1, file_to_write);
    fwrite(&svm->accuracy_c1, sizeof(double), 1, file_to_write);
    fwrite(&svm->accuracy_c2, sizeof(double), 1, file_to_write);
    fwrite(&svm->correct_c1, sizeof(int), 1, file_to_write);
    fwrite(&svm->correct_c2, sizeof(int), 1, file_to_write);

    //params
    fwrite(&svm->params, sizeof(double) * 4, 1, file_to_write);
    fwrite(&svm->verbose, sizeof(bool), 1, file_to_write);


    // functions
    fwrite(&svm->kernel_type, sizeof(char), 1, file_to_write);

    fclose(file_to_write);

    return 0; // no problems
}

/**
 * Support for deserialization
 */

int read_svm(Kernel_SVM *svm, const std::string &path) {
    FILE *file_to_read;
    file_to_read = fopen(path.c_str(), "rb");

    if (!file_to_read) {
        std::cout << "Error opening file. Reading was not possible !";
        return 1;
    }

    // read sizes
    fread(&svm->arr_xs_row_size, sizeof(int), 1, file_to_read);
    fread(&svm->arr_xs_column_size, sizeof(int), 1, file_to_read);
    fread(&svm->arr_xs_in_row_size, sizeof(int), 1, file_to_read);

    svm->arr_xs = (double *) malloc(svm->arr_xs_row_size * svm->arr_xs_column_size * sizeof(double));
    svm->arr_ys = (int *) malloc(svm->arr_xs_row_size * sizeof(int));
    svm->arr_alpha_s = (double *) malloc(svm->arr_xs_row_size * sizeof(double));
    svm->arr_xs_in = (double *) malloc(svm->arr_xs_in_row_size * svm->arr_xs_column_size * sizeof(double));
    svm->arr_ys_in = (int *) malloc(svm->arr_xs_in_row_size * sizeof(int));
    svm->arr_alpha_s_in = (double *) malloc(svm->arr_xs_in_row_size * sizeof(double));

    fread(svm->arr_xs, sizeof(double) * svm->arr_xs_row_size * svm->arr_xs_column_size, 1, file_to_read);
    fread(svm->arr_ys, sizeof(int) * svm->arr_xs_row_size, 1, file_to_read);
    fread(svm->arr_alpha_s, sizeof(double) * svm->arr_xs_row_size, 1, file_to_read);
    fread(svm->arr_xs_in, sizeof(double) * svm->arr_xs_in_row_size * svm->arr_xs_column_size, 1, file_to_read);
    fread(svm->arr_ys_in, sizeof(int) * svm->arr_xs_in_row_size, 1, file_to_read);
    fread(svm->arr_alpha_s_in, sizeof(double) * svm->arr_xs_in_row_size, 1, file_to_read);

    // b
    fread(&svm->b, sizeof(double), 1, file_to_read);

    //accuracies
    fread(&svm->accuracy, sizeof(double), 1, file_to_read);
    fread(&svm->accuracy_c1, sizeof(double), 1, file_to_read);
    fread(&svm->accuracy_c2, sizeof(double), 1, file_to_read);
    fread(&svm->correct_c1, sizeof(int), 1, file_to_read);
    fread(&svm->correct_c2, sizeof(int), 1, file_to_read);

    //params
    fread(&svm->params, sizeof(double) * 4, 1, file_to_read);
    fread(&svm->verbose, sizeof(bool), 1, file_to_read);

    // functions
    char k;
    fread(&k, sizeof(char), 1, file_to_read);
    set_kernel_function(svm, k);

    fclose(file_to_read);
    return 0;

}

void log(const std::string &str) {

    std::cout << str << " " << std::flush;

}


/**
 * Function f()
 */

double f(Kernel_SVM *svm, double *x) {

    int i;
    double ans;

    ans = 0.0;
    double xi[svm->arr_xs_column_size];
    for (i = 0; i < svm->arr_xs_row_size; i++) {
        get_row(svm->arr_xs, i, svm->arr_xs_column_size, xi);
        ans += svm->arr_alpha_s[i] * svm->arr_ys[i] * svm->K(xi, x, svm->arr_xs_column_size, svm->params);
    }
    for (i = 0; i < svm->arr_xs_in_row_size; i++) {
        get_row(svm->arr_xs_in, i, svm->arr_xs_column_size, xi);
        ans += svm->arr_alpha_s_in[i] * svm->arr_ys_in[i] * svm->K(xi, x, svm->arr_xs_column_size, svm->params);
    }
    ans += svm->b;

    return ans;
}


/**
 * Function g()
 */

double g(Kernel_SVM *svm, double *x) {

    double fx;
    int gx;

    fx = f(svm, x);
    if (fx >= 0.0) {
        gx = 1;
    } else {
        gx = -1;
    }

    return gx;
}


/**
 * Function f() parallel
 */

double f_parallel(Kernel_SVM *svm, double *x, int process_offset, int available_processes) {

    // Get the rank of the process
    int current_process;
    MPI_Comm_rank(MPI_COMM_WORLD, &current_process);

    int i;
    double ans = 0.0;
    double xi[svm->arr_xs_column_size];

    int iters_per_process = (int) (ceil((double) (svm->arr_xs_row_size) / (double) (available_processes)));

    int start = (current_process - process_offset) * iters_per_process;
    int end = start + iters_per_process;

    double local_ans = 0.0;

    for (i = start; (i < end) && (i < svm->arr_xs_row_size); i++) {
        get_row(svm->arr_xs, i, svm->arr_xs_column_size, xi);
        local_ans += svm->arr_alpha_s[i] * svm->arr_ys[i] * svm->K(xi, x, svm->arr_xs_column_size, svm->params);
    }

    for (i = start; (i < end) && (i < svm->arr_xs_row_size); i++) {
        get_row(svm->arr_xs_in, i, svm->arr_xs_column_size, xi);
        local_ans += svm->arr_alpha_s_in[i] * svm->arr_ys_in[i] * svm->K(xi, x, svm->arr_xs_column_size, svm->params);
    }

    MPI_Allreduce(&local_ans, &ans, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    ans += svm->b;

    return ans;
}


/**
 * Function g() parallel
 */

double g_parallel(Kernel_SVM *svm, double *x, int process_offset, int available_processes) {

    double fx;
    int gx;

    fx = f_parallel(svm, x, process_offset, available_processes);
    if (fx >= 0.0) {
        gx = 1;
    } else {
        gx = -1;
    }

    return gx;
}

#endif //HPC2022_SVM_H
