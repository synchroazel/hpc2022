#include <iostream>
#include <string>
#include <cmath>
#include <iomanip>
#include "mpi.h"
#include "svm.h"
#include "Dataset.h"
#include "utils.h"

#define DEBUG_SUPPORT_VECTORS false
#define MASTER_PROCESS 0
#define DEBUG_TRAIN false

/**
 * Linear kernel function
 */

double kernel::linear(const double *x1,
                      const double *x2,
                      size_t size,
                      double params[4] = nullptr) { // cost, gamma, coef0, degree, in this case none

    size_t i;
    double ans = 0.0;

    // u*v
    for (i = 0; i < size; i++) {
        //        x1   *    x2
        ans += x1[i] * x2[i];
    }

    return ans;
}


/**
 * Polynomial kernel function
 */

double kernel::polynomial(const double *x1,
                          const double *x2,
                          size_t size,
                          double params[4]) { // cost, gamma, coef0, degree

    size_t i;
    double ans = 0.0;

    //(gamma*u'*v + coef0)^degree
    for (i = 0; i < size; i++) {
        //      gamma    *    x1     *     x2
        ans += params[1] * x1[i] * x2[i];
    }
    ans += params[2]; // + coef0

    ans = std::pow(ans, params[3]); // ^ degree

    return ans;

}


/**
 * Radial kernel function
 */

double kernel::radial(const double *x1,
                      const double *x2,
                      size_t size,
                      const double params[4]) { // cost, gamma, coef0, degree

    size_t i;
    double ans = 0.0;

    //  exp(-gamma*|u-v|^2)
    for (i = 0; i < size; i++) {
        //      (u' - v)^2
        ans += (x1[i] - x2[i]) * (x1[i] - x2[i]);
    }
    //         exp(gamma * ...)
    ans = std::exp(-params[1] * ans);

    return ans;

}


/**
 * Sigmoid kernel function
 */

double kernel::sigmoid(const double *x1,
                       const double *x2,
                       size_t size,
                       const double params[4]) { // cost, gamma, coef0, degree

    size_t i;
    double ans = 0.0;

    //  tanh(gamma*u'*v + coef0)
    for (i = 0; i < size; i++) {
        //      gamma    *    u'     *     v
        ans += params[1] * x1[i] * x2[i];
    }
    ans += params[2]; // + coef0
    //         tanh(...)
    ans = std::tanh(ans);

    return ans;

}

/**
 * Serial Training function
 */

// training data must have only two classes
void serial_train(const Dataset &training_data,
                  Kernel_SVM *svm,
                  double hyper_parameters[4],
                  const double lr,
                  const double limit,
                  bool save_svm_flag = true,
                  const std::string &svm_save_dir_path = "",
                  size_t class_1 = 0,
                  const double eps = 0.0000001) {


    svm->params[0] = hyper_parameters[0];
    svm->params[1] = hyper_parameters[1];
    svm->params[2] = hyper_parameters[2];
    svm->params[3] = hyper_parameters[3];
    int y[training_data.rows_number];  // array
    memset(y, 0, training_data.rows_number * sizeof(int));
    size_t i = 0;
    //TODO: all checks
    int control = 0; // used for checks

    for (; i < training_data.rows_number; i++) {

        // look at y
        if (training_data.class_vector[i] == training_data.unique_classes[class_1]) { // if first class record
            y[i] = -1;  // -1
        } else { // if second class record

            y[i] = 1;  // +1

        }
    }

    size_t N;
    N = i; // number of rows

    i = 0; // i will be useful later
    size_t j;

    bool judge;
    double item1, item2, item3;
    double delta;
    double beta;
    double error;

    // set Lagrange Multiplier and Parameters
    double alpha[N];
    memset(alpha, 0, N * sizeof(double));

    beta = 1.0;

    // tmp rows
    double xi[training_data.predictors_column_number];
    memset(alpha, 0, training_data.predictors_column_number * sizeof(double));
    double xj[training_data.predictors_column_number];
    memset(alpha, 0, training_data.predictors_column_number * sizeof(double));

    // training
    if (svm->verbose) {
        std::cout << "\n┌────────────────────── Training ───────────────────────┐" << std::endl;
    }


    do {

        judge = false;
        error = 0.0;

        // update Alpha
        for (i = 0; i < N; i++) {

            // compute the partial derivative with respect to alpha

            item1 = 0;
            for (j = 0; j < N; j++) {
                get_row(training_data, i, false, xi);
                get_row(training_data, j, false, xj);
                item1 += alpha[j] * (double) y[i] * (double) y[j] *
                         svm->K(xi, xj, training_data.predictors_column_number, hyper_parameters);
            }

            // set item 2
            item2 = 0;

            for (j = 0; j < N; j++) {
                item2 += alpha[j] * (double) y[i] * (double) y[j];
            }

            // set such partial derivative to Delta

            delta = 1.0 - item1 - beta * item2;

            // update
            alpha[i] += lr * delta;
            if (alpha[i] < 0.0) {
                alpha[i] = 0.0;
            } else if (alpha[i] > hyper_parameters[0]/*Cost*/) {
                alpha[i] = hyper_parameters[0];
            } else if (std::abs(delta) > limit) {
                judge = true;
                error += std::abs(delta) - limit;
            }

        }

        // update bias Beta
        item3 = 0.0;
        for (i = 0; i < N; i++) {
            item3 += alpha[i] * (double) y[i];
        }
        beta += item3 * item3 / 2.0;

        // output Residual Error
        if (svm->verbose) {
            std::cout << "\r error = " << error << std::flush;
        }

    } while (judge);


    /* ----------------------------------------------------------------------------- */


    if (svm->verbose) {
        std::cout << "\n├───────────────────────────────────────────────────────┤" << std::endl;
    }

    // initialize, then realloc
    // NB: N are the rows of the new matrix
    // need to check for memory leaks, we will try another approach in the meanwhile

    svm->arr_xs = (double *) calloc(N * training_data.predictors_column_number, sizeof(double)); // matrix
    svm->arr_ys = (int *) calloc(N, sizeof(int));
    svm->arr_alpha_s = (double *) calloc(N, sizeof(double));

    svm->arr_xs_row_size = 0;
    svm->arr_xs_column_size = training_data.predictors_column_number;

    svm->arr_xs_in = (double *) calloc(N * training_data.predictors_column_number, sizeof(double)); // matrix
    svm->arr_ys_in = (int *) calloc(N, sizeof(int));
    svm->arr_alpha_s_in = (double *) calloc(N, sizeof(double));

    svm->arr_xs_in_row_size = 0;

    int sv = 0, svi = 0;
    for (i = 0; i < N; i++) {
        if ((eps < alpha[i]) && (alpha[i] < hyper_parameters[0]/*cost*/ - eps)) {
            // support vectors outside the margin
            get_row(training_data, i, false, xi);
            memcpy(svm->arr_xs + index(sv, 0, training_data.predictors_column_number), xi,
                   training_data.predictors_column_number * sizeof(double));
            ++svm->arr_xs_row_size;

            *(svm->arr_ys + sv) = y[i];

            *(svm->arr_alpha_s + sv) = alpha[i];

            ++sv;

        } else if (alpha[i] >= hyper_parameters[0]/*cost*/ - eps) {
            // support vectors inside the margin
            get_row(training_data, i, false, xi);
            memcpy(svm->arr_xs_in + index(svi, 0, training_data.predictors_column_number), xi,
                   training_data.predictors_column_number * sizeof(double));
            ++svm->arr_xs_in_row_size;

            *(svm->arr_ys_in + svi) = y[i];

            *(svm->arr_alpha_s_in + svi) = alpha[i];

            ++svi;
        }
    }

#if DEBUG_SUPPORT_VECTORS

    std::cout << "Before realloc" << std::endl;

    std::cout << "x" << std::endl;
    print_matrix(svm->arr_xs, svm->arr_xs_row_size, svm->arr_xs_column_size);

    std::cout << "y" << std::endl;
    print_vector(svm->arr_ys, svm->arr_ys_size);

    std::cout << "alpha" << std::endl;
    print_vector(svm->arr_alpha_s, svm->arr_alpha_size);

    std::cout << "x in" << std::endl;
    print_matrix(svm->arr_xs_in, svm->arr_xs_in_row_size, svm->arr_xs_column_size);

    std::cout << "y in" << std::endl;
    print_vector(svm->arr_ys_in, svm->arr_ys_in_size);

    std::cout << "alpha in" << std::endl;
    print_vector(svm->arr_alpha_s_in, svm->arr_alpha_in_size);

#endif
    // realloc to cut off extra 0s
    // TODO: check if realloc does bogus stuff

//    svm->arr_xs = (double *) reallocarray(svm->arr_xs, svm->arr_xs_row_size * svm->arr_xs_column_size, sizeof (double ));
//    svm->arr_ys = (int *) reallocarray(svm->arr_ys, svm->arr_xs_row_size, sizeof (int));
//    svm->arr_alpha_s = (double *) reallocarray(svm->arr_alpha_s, svm->arr_alpha_size, sizeof (double ));
//    svm->arr_xs_in = (double *) reallocarray(svm->arr_xs_in, svm->arr_xs_in_row_size * svm->arr_xs_in_column_size, sizeof (double ));
//    svm->arr_ys_in = (int *) reallocarray(svm->arr_ys_in, svm->arr_xs_in_row_size, sizeof (int));
//    svm->arr_alpha_s_in = (double *) reallocarray(svm->arr_alpha_s_in, svm->arr_alpha_in_size, sizeof (double ));

    svm->arr_xs = (double *) realloc(svm->arr_xs, svm->arr_xs_row_size * svm->arr_xs_column_size * sizeof(double));
    svm->arr_ys = (int *) realloc(svm->arr_ys, svm->arr_xs_row_size * sizeof(int));
    svm->arr_alpha_s = (double *) realloc(svm->arr_alpha_s, svm->arr_xs_row_size * sizeof(double));
    svm->arr_xs_in = (double *) realloc(svm->arr_xs_in,
                                        svm->arr_xs_in_row_size * svm->arr_xs_column_size * sizeof(double));
    svm->arr_ys_in = (int *) realloc(svm->arr_ys_in, svm->arr_xs_in_row_size * sizeof(int));
    svm->arr_alpha_s_in = (double *) realloc(svm->arr_alpha_s_in, svm->arr_xs_in_row_size * sizeof(double));

    // Update the bias
    svm->b = 0.0;
    for (i = 0; i < svm->arr_xs_row_size; i++) {
        svm->b += (double) svm->arr_ys[i];
        for (j = 0; j < svm->arr_xs_row_size; j++) {
            get_row(svm->arr_xs, i, false, xi);
            get_row(svm->arr_xs, j, false, xj);
            svm->b -=
                    svm->arr_alpha_s[j] * (double) svm->arr_ys[j] *
                    svm->K(xj, xi, svm->arr_xs_column_size, svm->params);
        }
        for (j = 0; j < svm->arr_xs_in_row_size; j++) {
            get_row(svm->arr_xs_in, i, false, xi);
            get_row(svm->arr_xs_in, j, false, xj);
            svm->b -=
                    svm->arr_alpha_s_in[j] * (double) svm->arr_ys_in[j] *
                    svm->K(xj, xi, svm->arr_xs_column_size, svm->params);
        }
    }
    svm->b /= (double) (svm->arr_xs_row_size + svm->arr_xs_in_row_size);
    if (svm->verbose) {
        std::cout << " bias = " << svm->b << std::endl;
        std::cout << "└───────────────────────────────────────────────────────┘\n" << std::endl;
    }

#if DEBUG_SUPPORT_VECTORS

    //print svm->xs
    log("xs:\n");
    for (i = 0; i < svm->arr_xs_row_size; i++) {
        for (j = 0; j < svm->arr_xs_column_size; j++) {
            log(std::to_string(svm->arr_xs[index(i,j,svm->arr_xs_column_size)]));
        }
        log("\n");
    }

    //print svm->ys
    log("ys: ");
    for (i = 0; i < svm->arr_ys_size; i++) {
        log(std::to_string(svm->arr_ys[i]));
    }
    log("\n");

    //print svm->alpha_s
    log("alpha_s:");
    for (i = 0; i < svm->arr_alpha_size; i++) {
        log(std::to_string(svm->arr_alpha_s[i]));
    } log("\n");

    //print svm->xs_in
    log("xs_in:\n");
    for (i = 0; i < svm->arr_xs_in_row_size; i++) {
        for (j = 0; j < svm->arr_xs_in_column_size; j++) {
            log(std::to_string(svm->arr_xs_in[index(i,j,svm->arr_xs_in_column_size)]));
        }
        log(" | ");
    }

    //print svm->ys_in
    log("ys_in: ");
    for (i = 0; i < svm->arr_ys_in_size; i++) {
        log(std::to_string(svm->arr_ys_in[i]) + " ");
    } log("\n");

    //print svm->alpha_s_in
    log("alpha_s_in: ");
    for (i = 0; i < svm->arr_alpha_in_size; i++) {
        log(std::to_string(svm->arr_alpha_s_in[i]));
    } log("\n");

#endif

    if (svm->verbose) {

        logtime();
        std::cout << "Number of support vectors *on* margin) = " << svm->arr_xs_row_size << std::endl;
        logtime();
        std::cout << "Number of support vectors *inside* margin) = " << svm->arr_xs_in_row_size << "\n" << std::endl;
    }

    if (save_svm_flag) {
        /** Write svm to file **/
        std::string s;
        if (svm_save_dir_path.empty()) {
            s = "./";
        } else {
            s = svm_save_dir_path;
        }

        // If not exist, create directory to save the model params
        switch (svm->kernel_type) {
            case 'l': {
                s = s + "linear_C" + std::to_string(hyper_parameters[0]) + ".svm";
                break;
            }
            case 'r': {
                s = s + "radial" + std::string(1, svm->kernel_type) +
                    "_C" + std::to_string(hyper_parameters[0]) +
                    "_G" + std::to_string(hyper_parameters[1]) +
                    ".svm";
                break;
            }
            case 's': {
                s = s + "sigmoid" + std::string(1, svm->kernel_type) +
                    "_C" + std::to_string(hyper_parameters[0]) +
                    "_G" + std::to_string(hyper_parameters[1]) +
                    "_O" + std::to_string(hyper_parameters[2]) +
                    ".svm";
                break;
            }
            case 'p': {
                s = s + "polynomial" + std::string(1, svm->kernel_type) +
                    "_C" + std::to_string(hyper_parameters[0]) +
                    "_G" + std::to_string(hyper_parameters[1]) +
                    "_O" + std::to_string(hyper_parameters[2]) +
                    "_D" + std::to_string(hyper_parameters[3]) +
                    ".svm";
                break;
            }
        }

        save_svm(svm, s);

        logtime();
        std::cout << "Svm was saved as " << s << std::endl;
    }

    // TODO: capire di cosa fare il free

}


/**
 * Serial testing function
 */

void serial_test(Dataset test_data,
                 Kernel_SVM *svm,
                 size_t class_1 = 0) {

    // split all training data into class1 and class2 data

    auto *class1_data = (double *) calloc(test_data.rows_number * test_data.predictors_column_number, sizeof(double));
    auto *class2_data = (double *) calloc(test_data.rows_number * test_data.predictors_column_number, sizeof(double));

    auto *cur_row = (double *) calloc(test_data.predictors_column_number, sizeof(double));
    size_t c1 = 0, c2 = 0;

    for (size_t i = 0; i < test_data.rows_number; i++) {

        get_row(test_data, i, false, cur_row);

        if (test_data.class_vector[i] == test_data.unique_classes[class_1]) {
            memcpy(class1_data + (c1 * test_data.predictors_column_number), cur_row,
                   test_data.predictors_column_number * sizeof(double));
            ++c1;
        } else {
            memcpy(class2_data + (c2 * test_data.predictors_column_number), cur_row,
                   test_data.predictors_column_number * sizeof(double));
            ++c2;
        }
    }

    size_t i;
    int result = 0;

    svm->correct_c1 = 0;
    for (i = 0; i < c1; i++) {
        result = (int) g(svm, class1_data + index(i, 0, test_data.predictors_column_number));
        if (result == -1) {
            ++svm->correct_c1;
        }
    }

    svm->correct_c2 = 0;
    for (i = 0; i < c1; i++) {
        result = (int) g(svm, class2_data + index(i, 0, test_data.predictors_column_number));
        if (result == 1) {
            ++svm->correct_c2;
        }
    }

    svm->accuracy =
            (double) (svm->correct_c1 + svm->correct_c2) / (double) (c1 + c2);
    svm->accuracy_c1 = (double) svm->correct_c1 / (double) c1;
    svm->accuracy_c2 = (double) svm->correct_c2 / (double) c2;

    if (svm->verbose) {
        std::cout << "\n┌────────────── Test Results ───────────────┐" << std::endl;

        std::cout << "Cost: " << svm->params[0] << " | Gamma: " << svm->params[1] << " | Coef0: " << svm->params[2]
                  << " | Degree: " << svm->params[3] << std::endl;
        std::cout << "  accuracy-all:\t\t" << std::setprecision(6) << svm->accuracy << " ("
                  << svm->correct_c1 + svm->correct_c2 << "/" << c1 + c2 << " hits)" << std::endl;
        std::cout << "  accuracy-class1:\t" << std::setprecision(6) << svm->accuracy_c1 << " (" << svm->correct_c1
                  << "/"
                  << c1 << " hits)" << std::endl;
        std::cout << "  accuracy-class2:\t" << std::setprecision(6) << svm->accuracy_c2 << " (" << svm->correct_c2
                  << "/"
                  << c2 << " hits)" << std::endl;
        std::cout << "└───────────────────────────────────────────┘" << std::endl;
    }


    free(class1_data);
    free(class2_data);
    free(cur_row);

}


/**
 * Parallel Training function
 */

// training data must have only two classes
void parallel_train(const Dataset &training_data,
                    Kernel_SVM *svm,
                    double hyper_parameters[4],
                    const double lr,
                    const double limit,
                    int process_offset,
                    int available_processes,
                    bool save_svm_flag = true,
                    const std::string &svm_save_dir_path = "",
                    size_t class_1 = 0,
                    const double eps = 0.0000001) {


    // Get the rank of the process
    int current_process;
    MPI_Comm_rank(MPI_COMM_WORLD, &current_process);
    int iters_per_process = (int) (ceil((double) (training_data.rows_number) / (double) (available_processes)));
#if DEBUG_TRAIN
    if (current_process == process_offset) {
        std::cout << "Number of rows: " << training_data.rows_number << std::endl;
        std::cout << "Number of processes: " << available_processes << std::endl;
        std::cout << "Iters per process: " << iters_per_process << std::endl;
    }
#endif


    // TODO: implement parallelization
    svm->params[0] = hyper_parameters[0];
    svm->params[1] = hyper_parameters[1];
    svm->params[2] = hyper_parameters[2];
    svm->params[3] = hyper_parameters[3];

    //auto* local_y = (int* ) calloc()
    int y[training_data.rows_number];  // array
    memset(y, 0, training_data.rows_number * sizeof(int));
    size_t i;
    //TODO: all checks
    int control = 0; // used for checks

    int start = (current_process - process_offset) * iters_per_process;
    int end = start + iters_per_process;
    for (i = start; i < end; i++) {

        // look at y
        if (training_data.class_vector[i] == training_data.unique_classes[class_1]) { // if first class record
            y[i] = -1;  // -1
        } else { // if second class record

            y[i] = 1;  // +1

        }
    }

    size_t N;
    // N = i; // number of rows
    N = training_data.rows_number;

    i = 0; // i will be useful later
    size_t j;

    bool judge;
    double item1 = 0, item2 = 0, item3 = 0;
    double alpha[N];
    memset(alpha, 0, N * sizeof(double));
    double delta;
    double beta;
    double error;

    // set Lagrange Multiplier and Parameters
    double local_alpha[N];
    memset(local_alpha, 0, N * sizeof(double));

    beta = 1.0;

    // tmp rows
    double xi[training_data.predictors_column_number];
    memset(local_alpha, 0, training_data.predictors_column_number * sizeof(double));
    double xj[training_data.predictors_column_number];
    memset(local_alpha, 0, training_data.predictors_column_number * sizeof(double));

    // training
    if (svm->verbose) {
        std::cout << "\n┌────────────────────── Training ───────────────────────┐" << std::endl;
    }


    /* ----------------------------------------------------------------------------- */


    double item1_local;
    double item2_local;
    double item3_local;

    // std::cout << "current process: " << current_process << " process offset: " << process_offset << " N: " << N << std::endl;

    do {

        judge = false;
        error = 0.0;

        // update Alpha
        for (i = start; (i < end) && (i < N); i++) {

            item1_local = 0;
            for (j = 0; j < N; j++) {  // OMP?
                get_row(training_data, i, false, xi);
                get_row(training_data, j, false, xj);
                item1_local += local_alpha[j] * (double) y[i] * (double) y[j] *
                               svm->K(xi, xj, training_data.predictors_column_number, hyper_parameters);
            }

            MPI_Allreduce(&item1_local, &item1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

            item2_local = 0;
            for (j = 0; j < N; j++) {  // OMP?
                item2_local += local_alpha[j] * (double) y[i] * (double) y[j];
            }

            MPI_Allreduce(&item2_local, &item2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

            delta = 1.0 - item1 - beta * item2;

            local_alpha[i] += lr * delta;

            if (local_alpha[i] < 0.0) {
                local_alpha[i] = 0.0;
            } else if (local_alpha[i] > hyper_parameters[0]/*Cost*/) {
                local_alpha[i] = hyper_parameters[0];
            } else if (std::abs(delta) > limit) {
                judge = true;
                error += std::abs(delta) - limit;
            }

        }

#if DEBUG_TRAIN
        std::cout << "\nlocal alpha on process " << current_process << ": " << std::endl;
        for (int k = 0; k < sizeof(local_alpha) / sizeof(double); k++) {
            std::cout << local_alpha[k] << " ";
        } std::cout << std::endl;
#endif

        MPI_Allreduce(&local_alpha, &alpha, N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

#if DEBUG_TRAIN
        if (current_process == MASTER_PROCESS) {
            std::cout << "\n" << std::endl;
        }
        std::cout << "alpha on process " << current_process << " has length " << sizeof(alpha) / sizeof(double) << std::endl;
#endif

        // update bias Beta
        item3_local = 0.0;
        for (i = start; (i < end) && (i < N); i++) {
            item3_local += alpha[i] * (double) y[i];
        }

        MPI_Allreduce(&item3_local, &item3, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        beta += item3 * item3 / 2.0;

        // output Residual Error
        if (svm->verbose) {
            std::cout << "\r error = " << error << std::flush;
        }


    } while (judge);


    /* ----------------------------------------------------------------------------- */


    if (svm->verbose) { std::cout << "\n├───────────────────────────────────────────────────────┤" << std::endl; }

    // initialize, then realloc
    // NB: N are the rows of the new matrix
    // need to check for memory leaks, we will try another approach in the meanwhile

    auto *local_arr_xs = (double *) calloc(iters_per_process * training_data.predictors_column_number, sizeof(double));
    auto *local_arr_ys = (int *) calloc(iters_per_process, sizeof(int));
    auto *local_arr_alpha_s = (double *) calloc(iters_per_process, sizeof(double));
    int local_arr_xs_row_size = 0; //
    auto *local_arr_xs_in = (double *) calloc(iters_per_process * training_data.predictors_column_number,sizeof(double));
    auto *local_arr_ys_in = (int *) calloc(iters_per_process, sizeof(int));
    auto *local_arr_alpha_s_in = (double *) calloc(iters_per_process, sizeof(double));
    int local_arr_xs_in_row_size = 0; //

    int *local_xs_sizes = (int *) calloc(iters_per_process, sizeof(int));
    int *local_xs_in_sizes = (int *) calloc(iters_per_process, sizeof(int));

    //int sv = 0, svi = 0;

    for (i = start; (i < end) && (i < N); i++) {

        if ((eps < alpha[i]) && (alpha[i] < hyper_parameters[0]/*cost*/ - eps)) {
            // support vectors outside the margin
            get_row(training_data, i, false, xi);
            memcpy(local_arr_xs + index(local_arr_xs_row_size, 0, training_data.predictors_column_number), xi,
                   training_data.predictors_column_number * sizeof(double));
            ++local_arr_xs_row_size;

            *(local_arr_ys + local_arr_xs_row_size) = y[i];

            *(local_arr_alpha_s + local_arr_xs_row_size) = alpha[i];

            //++sv;

        } else if (alpha[i] >= hyper_parameters[0]/*cost*/ - eps) {
            // support vectors inside the margin
            get_row(training_data, i, false, xi);
            memcpy(local_arr_xs_in + index(local_arr_xs_in_row_size, 0, training_data.predictors_column_number), xi,
                   training_data.predictors_column_number * sizeof(double));
            ++local_arr_xs_in_row_size;

            *(local_arr_ys_in + local_arr_xs_in_row_size) = y[i];

            *(local_arr_alpha_s_in + local_arr_xs_in_row_size) = alpha[i];

            //++svi;
        }
    }

    local_xs_sizes[current_process] = local_arr_xs_row_size;
    local_xs_in_sizes[current_process] = local_arr_xs_row_size;

    MPI_Reduce(&local_arr_xs_row_size, &svm->arr_xs_row_size, 1, MPI_INT, MPI_SUM, process_offset, MPI_COMM_WORLD);
    MPI_Reduce(&local_arr_xs_in_row_size, &svm->arr_xs_in_row_size, 1, MPI_INT, MPI_SUM, process_offset, MPI_COMM_WORLD);

    if (current_process == process_offset) {
        svm->arr_xs = (double *) calloc(svm->arr_xs_row_size * training_data.predictors_column_number, sizeof(double));
        svm->arr_ys = (int *) calloc(svm->arr_xs_row_size, sizeof(int));
        svm->arr_alpha_s = (double *) calloc(svm->arr_xs_row_size, sizeof(double));
        svm->arr_xs_in = (double *) calloc(svm->arr_xs_in_row_size * training_data.predictors_column_number,sizeof(double));
        svm->arr_ys_in = (int *) calloc(svm->arr_xs_in_row_size, sizeof(int));
        svm->arr_alpha_s_in = (double *) calloc(svm->arr_xs_in_row_size, sizeof(double));
    }

    // TODO : implement a gather for all_sizes, and check the below

//    svm->arr_xs = (double *) calloc(N * training_data.predictors_column_number, sizeof(double)); // matrix
//    svm->arr_ys = (int *) calloc(N, sizeof(int));
//    svm->arr_alpha_s = (double *) calloc(N, sizeof(double));
//    svm->arr_xs_row_size = 0;
//    svm->arr_xs_column_size = training_data.predictors_column_number;
//    svm->arr_xs_in = (double *) calloc(N * training_data.predictors_column_number, sizeof(double)); // matrix
//    svm->arr_ys_in = (int *) calloc(N, sizeof(int));
//    svm->arr_alpha_s_in = (double *) calloc(N, sizeof(double));
//    svm->arr_xs_in_row_size = 0;

#if DEBUG_SUPPORT_VECTORS

    std::cout << "Before realloc" << std::endl;

    std::cout << "x" << std::endl;
    print_matrix(svm->arr_xs, svm->arr_xs_row_size, svm->arr_xs_column_size);

    std::cout << "y" << std::endl;
    print_vector(svm->arr_ys, svm->arr_ys_size);

    std::cout << "alpha" << std::endl;
    print_vector(svm->arr_alpha_s, svm->arr_alpha_size);

    std::cout << "x in" << std::endl;
    print_matrix(svm->arr_xs_in, svm->arr_xs_in_row_size, svm->arr_xs_column_size);

    std::cout << "y in" << std::endl;
    print_vector(svm->arr_ys_in, svm->arr_ys_in_size);

    std::cout << "alpha in" << std::endl;
    print_vector(svm->arr_alpha_s_in, svm->arr_alpha_in_size);

#endif

    // realloc to cut off extra 0s
    // TODO: check if realloc does bogus stuff

//    svm->arr_xs = (double *) reallocarray(svm->arr_xs, svm->arr_xs_row_size * svm->arr_xs_column_size, sizeof (double ));
//    svm->arr_ys = (int *) reallocarray(svm->arr_ys, svm->arr_xs_row_size, sizeof (int));
//    svm->arr_alpha_s = (double *) reallocarray(svm->arr_alpha_s, svm->arr_alpha_size, sizeof (double ));
//
//    svm->arr_xs_in = (double *) reallocarray(svm->arr_xs_in, svm->arr_xs_in_row_size * svm->arr_xs_in_column_size, sizeof (double ));
//    svm->arr_ys_in = (int *) reallocarray(svm->arr_ys_in, svm->arr_xs_in_row_size, sizeof (int));
//    svm->arr_alpha_s_in = (double *) reallocarray(svm->arr_alpha_s_in, svm->arr_alpha_in_size, sizeof (double ));

    svm->arr_xs = (double *) realloc(svm->arr_xs, svm->arr_xs_row_size * svm->arr_xs_column_size * sizeof(double));
    svm->arr_ys = (int *) realloc(svm->arr_ys, svm->arr_xs_row_size * sizeof(int));
    svm->arr_ys_in = (int *) realloc(svm->arr_ys_in, svm->arr_xs_in_row_size * sizeof(int));
    svm->arr_alpha_s_in = (double *) realloc(svm->arr_alpha_s_in, svm->arr_xs_in_row_size * sizeof(double));

    // Update the bias
    svm->b = 0.0;
    for (i = 0; i < svm->arr_xs_row_size; i++) {
        svm->b += (double) svm->arr_ys[i];
        for (j = 0; j < svm->arr_xs_row_size; j++) {
            get_row(svm->arr_xs, i, false, xi);
            get_row(svm->arr_xs, j, false, xj);
            svm->b -=
                    svm->arr_alpha_s[j] * (double) svm->arr_ys[j] *
                    svm->K(xj, xi, svm->arr_xs_column_size, svm->params);
        }
        for (j = 0; j < svm->arr_xs_in_row_size; j++) {
            get_row(svm->arr_xs_in, i, false, xi);
            get_row(svm->arr_xs_in, j, false, xj);
            svm->b -=
                    svm->arr_alpha_s_in[j] * (double) svm->arr_ys_in[j] *
                    svm->K(xj, xi, svm->arr_xs_column_size, svm->params);
        }
    }
    svm->b /= (double) (svm->arr_xs_row_size + svm->arr_xs_in_row_size);
    if (svm->verbose) {
        std::cout << " bias = " << svm->b << std::endl;
        std::cout << "└───────────────────────────────────────────────────────┘\n" << std::endl;
    }

#if DEBUG_SUPPORT_VECTORS

    //print svm->xs
    log("xs:\n");
    for (i = 0; i < svm->arr_xs_row_size; i++) {
        for (j = 0; j < svm->arr_xs_column_size; j++) {
            log(std::to_string(svm->arr_xs[index(i,j,svm->arr_xs_column_size)]));
        }
        log("\n");
    }

    //print svm->ys
    log("ys: ");
    for (i = 0; i < svm->arr_ys_size; i++) {
        log(std::to_string(svm->arr_ys[i]));
    }
    log("\n");

    //print svm->alpha_s
    log("alpha_s:");
    for (i = 0; i < svm->arr_alpha_size; i++) {
        log(std::to_string(svm->arr_alpha_s[i]));
    } log("\n");

    //print svm->xs_in
    log("xs_in:\n");
    for (i = 0; i < svm->arr_xs_in_row_size; i++) {
        for (j = 0; j < svm->arr_xs_in_column_size; j++) {
            log(std::to_string(svm->arr_xs_in[index(i,j,svm->arr_xs_in_column_size)]));
        }
        log(" | ");
    }

    //print svm->ys_in
    log("ys_in: ");
    for (i = 0; i < svm->arr_ys_in_size; i++) {
        log(std::to_string(svm->arr_ys_in[i]) + " ");
    } log("\n");

    //print svm->alpha_s_in
    log("alpha_s_in: ");
    for (i = 0; i < svm->arr_alpha_in_size; i++) {
        log(std::to_string(svm->arr_alpha_s_in[i]));
    } log("\n");

#endif

    if (svm->verbose) {

        logtime();
        std::cout << "Number of support vectors *on* margin) = " << svm->arr_xs_row_size << std::endl;
        logtime();
        std::cout << "Number of support vectors *inside* margin) = " << svm->arr_xs_in_row_size << "\n" << std::endl;
    }

    if ((save_svm_flag) & (current_process == MASTER_PROCESS)) {

        /** Write svm to file */

        std::string s;
        if (svm_save_dir_path.empty()) {
            s = "./";
        } else {
            s = svm_save_dir_path;
        }

        // If not exist, create directory to save the model params
        switch (svm->kernel_type) {
            case 'l': {
                s = s + "linear_C" + std::to_string(hyper_parameters[0]) + ".svm";
                break;
            }
            case 'r': {
                s = s + "radial" + std::string(1, svm->kernel_type) +
                    "_C" + std::to_string(hyper_parameters[0]) +
                    "_G" + std::to_string(hyper_parameters[1]) +
                    ".svm";
                break;
            }
            case 's': {
                s = s + "sigmoid" + std::string(1, svm->kernel_type) +
                    "_C" + std::to_string(hyper_parameters[0]) +
                    "_G" + std::to_string(hyper_parameters[1]) +
                    "_O" + std::to_string(hyper_parameters[2]) +
                    ".svm";
                break;
            }
            case 'p': {
                s = s + "polynomial" + std::string(1, svm->kernel_type) +
                    "_C" + std::to_string(hyper_parameters[0]) +
                    "_G" + std::to_string(hyper_parameters[1]) +
                    "_O" + std::to_string(hyper_parameters[2]) +
                    "_D" + std::to_string(hyper_parameters[3]) +
                    ".svm";
                break;
            }
        }

        save_svm(svm, s);

        logtime();
        std::cout << "Svm was saved as " << s << std::endl;

    }

    // TODO: capire di cosa fare il free

}

/**
 * Parallel testing function
 */

void parallel_test(Dataset test_data,
                   Kernel_SVM *svm,
                   int process_offset,
                   int available_processes,
                   size_t class_1 = 0) {

    // todo: implement parallel logic

    // Get the rank of the process
    int process_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);


    auto *class1_data = (double *) calloc(test_data.rows_number * test_data.predictors_column_number,
                                          sizeof(double));
    auto *class2_data = (double *) calloc(test_data.rows_number * test_data.predictors_column_number,
                                          sizeof(double));

    auto *cur_row = (double *) calloc(test_data.predictors_column_number, sizeof(double));
    size_t c1 = 0, c2 = 0;

    for (size_t i = 0; i < test_data.rows_number; i++) {

        get_row(test_data, i, false, cur_row);

        if (test_data.class_vector[i] == test_data.unique_classes[class_1]) {
            memcpy(class1_data + (c1 * test_data.predictors_column_number), cur_row,
                   test_data.predictors_column_number * sizeof(double));
            ++c1;
        } else {
            memcpy(class2_data + (c2 * test_data.predictors_column_number), cur_row,
                   test_data.predictors_column_number * sizeof(double));
            ++c2;
        }
    }

    size_t i;
    int result = 0;

    svm->correct_c1 = 0;
    for (i = 0; i < c1; i++) {
        result = (int) g(svm, class1_data + index(i, 0, test_data.predictors_column_number));
        if (result == -1) {
            ++svm->correct_c1;
        }
    }

    svm->correct_c2 = 0;
    for (i = 0; i < c1; i++) {
        result = (int) g(svm, class2_data + index(i, 0, test_data.predictors_column_number));
        if (result == 1) {
            ++svm->correct_c2;
        }
    }

    svm->accuracy =
            (double) (svm->correct_c1 + svm->correct_c2) / (double) (c1 + c2);
    svm->accuracy_c1 = (double) svm->correct_c1 / (double) c1;
    svm->accuracy_c2 = (double) svm->correct_c2 / (double) c2;

    if (svm->verbose) {
        std::cout << "\n┌───────────────── Test Results ──────────────────┐" << std::endl;

        std::cout << "  Cost: " << svm->params[0] << " | Gamma: " << svm->params[1] << " | Coef0: "
                  << svm->params[2] << " | Degree: " << svm->params[3] << std::endl;
        std::cout << "  accuracy-all:\t\t" << std::setprecision(6) << svm->accuracy << " ("
                  << svm->correct_c1 + svm->correct_c2 << "/" << c1 + c2 << " hits)" << std::endl;
        std::cout << "  accuracy-class1:\t" << std::setprecision(6) << svm->accuracy_c1 << " (" << svm->correct_c1
                  << "/"
                  << c1 << " hits)" << std::endl;
        std::cout << "  accuracy-class2:\t" << std::setprecision(6) << svm->accuracy_c2 << " (" << svm->correct_c2
                  << "/"
                  << c2 << " hits)" << std::endl;
        std::cout << "└─────────────────────────────────────────────────┘\n" << std::endl;
    }

    free(class1_data);
    free(class2_data);
    free(cur_row);

}


