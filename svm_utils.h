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
#define FINAL_PRINT false


/**
 * Linear kernel function
 */

double kernel::linear(const double *x1,
                      const double *x2,
                      int size,
                      double params[4] = nullptr) { // cost, gamma, coef0, degree, in this case none

    int i;
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
                          int size,
                          double params[4]) { // cost, gamma, coef0, degree

    int i;
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
                      int size,
                      const double params[4]) { // cost, gamma, coef0, degree

    int i;
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
                       int size,
                       const double params[4]) { // cost, gamma, coef0, degree

    int i;
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
                  int class_1 = 0,
                  const double eps = 0.0000001) {


    svm->params[0] = hyper_parameters[0];
    svm->params[1] = hyper_parameters[1];
    svm->params[2] = hyper_parameters[2];
    svm->params[3] = hyper_parameters[3];
    int y[training_data.rows_number];  // array
    memset(y, 0, training_data.rows_number * sizeof(int));
    int i = 0;
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

    int N;
    N = i; // number of rows

    i = 0; // i will be useful later
    int j;

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
                         svm->K(xi, xj, (int)training_data.predictors_column_number, hyper_parameters);
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
    svm->arr_xs_column_size = (int)training_data.predictors_column_number;

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
    print_vector(svm->arr_ys, svm->arr_xs_row_size);

    std::cout << "alpha" << std::endl;
    print_vector(svm->arr_alpha_s, svm->arr_xs_row_size);

    std::cout << "x in" << std::endl;
    print_matrix(svm->arr_xs_in, svm->arr_xs_in_row_size, svm->arr_xs_column_size);

    std::cout << "y in" << std::endl;
    print_vector(svm->arr_ys_in, svm->arr_xs_in_row_size);

    std::cout << "alpha in" << std::endl;
    print_vector(svm->arr_alpha_s_in, svm->arr_xs_in_row_size);

#endif
    // realloc to cut off extra 0

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
    for (i = 0; i < svm->arr_xs_row_size; i++) {
        log(std::to_string(svm->arr_ys[i]));
    }
    log("\n");

    //print svm->alpha_s
    log("alpha_s:");
    for (i = 0; i < svm->arr_xs_row_size; i++) {
        log(std::to_string(svm->arr_alpha_s[i]));
    } log("\n");

    //print svm->xs_in
    log("xs_in:\n");
    for (i = 0; i < svm->arr_xs_in_row_size; i++) {
        for (j = 0; j < svm->arr_xs_column_size; j++) {
            log(std::to_string(svm->arr_xs_in[index(i,j,svm->arr_xs_column_size)]));
        }
        log(" | ");
    }

    //print svm->ys_in
    log("ys_in: ");
    for (i = 0; i < svm->arr_xs_row_size; i++) {
        log(std::to_string(svm->arr_ys_in[i]) + " ");
    } log("\n");

    //print svm->alpha_s_in
    log("alpha_s_in: ");
    for (i = 0; i < svm->arr_xs_row_size; i++) {
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
                 int class_1 = 0) {

    // split all training data into class1 and class2 data

    auto *class1_data = (double *) calloc(test_data.rows_number * test_data.predictors_column_number, sizeof(double));
    auto *class2_data = (double *) calloc(test_data.rows_number * test_data.predictors_column_number, sizeof(double));

    auto *cur_row = (double *) calloc(test_data.predictors_column_number, sizeof(double));
    int c1 = 0, c2 = 0;

    for (int i = 0; i < test_data.rows_number; i++) {

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

    int i;
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
// TODO: debug parallel
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
                    int class_1 = 0,
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

    // std::cout << "Current process "  << current_process << std::endl;

#endif

    int N = (int) training_data.rows_number;
    // TODO: implement parallelization
    svm->params[0] = hyper_parameters[0];

    svm->params[1] = hyper_parameters[1];

    svm->params[2] = hyper_parameters[2];

    svm->params[3] = hyper_parameters[3];

    //int y[training_data.rows_number];  // array
    //memset(y, 0, training_data.rows_number * sizeof(int));

    auto* y = (int*) calloc(N, sizeof(int));
    auto* local_y = (int*) calloc(N, sizeof(int));

    int i;
    //TODO: all checks
    int control = 0; // used for checks

    int start = (current_process - process_offset) * iters_per_process;
    int end = start + iters_per_process;
    for (i = start; i < end && i < N; i++) {

        // look at y
        if (training_data.class_vector[i] == training_data.unique_classes[class_1]) { // if first class record
            local_y[i] = -1;  // -1
        } else { // if second class record

            local_y[i] = 1;  // +1

        }
    }

    MPI_Allreduce(local_y, y,N, MPI_INT, MPI_SUM,MPI_COMM_WORLD);
    free(local_y);
    local_y= nullptr;




    i = 0; // i will be useful later
    int j;

    int judge=0;

    double item1 = 0, item2 = 0, item3 = 0;
    auto* alpha = (double*) calloc(N , sizeof(double));
    double delta=0;
    double beta=0;
    double error=0;

    // set Lagrange Multiplier and Parameters

    beta = 1.0;

    // tmp rows
    auto* xi = (double*) calloc(training_data.predictors_column_number, sizeof(double ));
    auto* xj = (double*) calloc(training_data.predictors_column_number, sizeof(double ));

    // training
    if (svm->verbose) {
        std::cout << "\n┌────────────────────── Training ───────────────────────┐" << std::endl;
    }


    /* ----------------------------------------------------------------------------- */



    double item1_local;
    double item2_local;
    double item3_local;
#if DEBUG_TRAIN
    std::cout << "\nprocess " << current_process << " starting point: " << start << " | ending point: " << end-1 << std::endl;
#endif

    do {

        judge = 0;
        error = 0.0;

        // update Alpha

        for (i = 0; i < N; i++) {

            item1_local = 0;
            item1=0;

            for (j = start; j < end && j < N; j++) {

                get_row(training_data, i, false, xi);
                get_row(training_data, j, false, xj);
                item1_local += alpha[j] * (double) y[i] * (double) y[j] *
                               svm->K(xi, xj, (int) training_data.predictors_column_number, hyper_parameters);
            }

            MPI_Allreduce(&item1_local, &item1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#if DEBUG_TRAIN
            // if(current_process == process_offset) std::cout << std::endl <<"final: " << item1 << "\n";
#endif
            item2_local = 0;
            item2=0;

            for (j = start; j < N && j<end; j++) {
                item2_local += alpha[j] * (double) y[i] * (double) y[j];
            }


            MPI_Allreduce(&item2_local, &item2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

            delta = 1.0 - item1 - beta * item2;

            alpha[i] += lr * delta;



            if (alpha[i] < 0.0) {
                alpha[i] = 0.0;
            } else if (alpha[i] > hyper_parameters[0]/*Cost*/) {
                alpha[i] = hyper_parameters[0];
            } else if (std::abs(delta) > limit) {
                ++judge;
                error += std::abs(delta) - limit;
            }
        }


#if DEBUG_TRAIN
        // std::cout << "\n[BEFORE REDUCE] local alpha on process " << current_process << ": " << std::endl;
        // for (int k = 0; k < N; k++) {
        //     std::cout << local_alpha[k] << " ";
        // } std::cout << std::endl;
#endif


#if DEBUG_TRAIN
        if(current_process == process_offset){
        std::cout << "\n[AFTER REDUCE] alpha on process " << current_process << ": " << std::endl;

            print_vector(alpha, N);
        }
#endif



        // update bias Beta
        item3_local = 0.0;
        item3=0;
        for (i = start; (i < end) && (i < N); i++) {
            item3_local += alpha[i] * (double) y[i];
        }
#if DEBUG_TRAIN
        // std::cout << "process " << current_process << " has item3_local = " << item3_local << "\n" << std::endl;
#endif

        MPI_Allreduce(&item3_local, &item3, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        beta += item3 * item3 / 2.0;

        // output Residual Error
        if (svm->verbose) {
            std::cout << "\r error = " << error << std::flush;
        }
#if DEBUG_TRAIN
        if(current_process == process_offset) std::cout << std::endl <<"judge: " << judge << "\n";
#endif
    } while (judge > 0);

    /* ----------------------------------------------------------------------------- */

#if DEBUG_TRAIN
    std::cout << "After do-while, process " << current_process << " is still alive" << std::endl;

#endif


    if (svm->verbose) { std::cout << "\n├───────────────────────────────────────────────────────┤" << std::endl; }

    // initialize, then realloc
    // NB: N are the rows of the new matrix
    // need to check for memory leaks, we will try another approach in the meanwhile



    // used for SVM kernel
    auto *local_arr_xs = (double *) calloc(iters_per_process * training_data.predictors_column_number, sizeof(double));
    auto *local_arr_ys = (int *) calloc(iters_per_process, sizeof(int));
    auto *local_arr_alpha_s = (double *) calloc(iters_per_process, sizeof(double));
    int local_arr_xs_row_size = 0; //
    auto *local_arr_xs_in = (double *) calloc(iters_per_process * training_data.predictors_column_number,sizeof(double));
    auto *local_arr_ys_in = (int *) calloc(iters_per_process, sizeof(int));
    auto *local_arr_alpha_s_in = (double *) calloc(iters_per_process, sizeof(double));
    int local_arr_xs_in_row_size = 0; //

    // used to pass sizes. One element for process (only for process 0)
    int *xs_sizes = (int *) calloc(available_processes, sizeof(int));
    int *xs_in_sizes = (int *) calloc(available_processes, sizeof(int));

    double local_bias = 0.0;

#if DEBUG_TRAIN
    // print out alpha
    if(current_process == process_offset){
        std::cout << "Process " << current_process << " printing alpha: " << std:: endl;
        print_vector(alpha, N);
    }
#endif


    //exit(1);

    for (i = start; (i < end) && (i < N); i++) {

        if ((eps < alpha[i]) && (alpha[i] < hyper_parameters[0]/*cost*/ - eps)) {

            // support vectors outside the margin
            get_row(training_data, i, false, xi);
            memcpy(local_arr_xs + index(local_arr_xs_row_size, 0, training_data.predictors_column_number), xi,
                   training_data.predictors_column_number * sizeof(double));
#if DEBUG_TRAIN
            std::cout << "process " << current_process << " found a sv outside "<< std::endl;
#endif
            *(local_arr_ys + local_arr_xs_row_size) = y[i];

            *(local_arr_alpha_s + local_arr_xs_row_size) = alpha[i];
            ++local_arr_xs_row_size;

            //++sv;

        } else if (alpha[i] >= hyper_parameters[0]/*cost*/ - eps) {

            // support vectors inside the margin
            get_row(training_data, i, false, xi);
            memcpy(local_arr_xs_in + index(local_arr_xs_in_row_size, 0, training_data.predictors_column_number), xi,
                   training_data.predictors_column_number * sizeof(double));
#if DEBUG_TRAIN
            std::cout << "process " << current_process << " found a sv inside "<< std::endl;
#endif
            *(local_arr_ys_in + local_arr_xs_in_row_size) = y[i];

            *(local_arr_alpha_s_in + local_arr_xs_in_row_size) = alpha[i];
            ++local_arr_xs_in_row_size;

        }


    }

#if DEBUG_TRAIN
    std::cout << "process " << current_process << " started at " << start << " and ended at " << end << std::endl;
    std::cout << "process " << current_process << " has " << local_arr_xs_row_size << " support vectors on the margin" << std::endl;
    std::cout << "process " << current_process << " has " << local_arr_xs_in_row_size << " support vectors inside the margin" << std::endl;
#endif
    // Update the bias

    for (i = 0; i < local_arr_xs_row_size; i++) {
        local_bias += (double) local_arr_ys[i];
        for (j = 0; j < local_arr_xs_row_size; j++) {
            get_row(local_arr_xs, i, false, xi);
            get_row(local_arr_xs, j, false, xj);
            local_bias -=
                    local_arr_alpha_s[j] * (double) local_arr_ys[j] *
                    svm->K(xj, xi, (int)training_data.predictors_column_number, hyper_parameters);
        }
        for (j = 0; j < local_arr_xs_in_row_size; j++) {
            get_row(local_arr_xs_in, i, false, xi);
            get_row(local_arr_xs_in, j, false, xj);
            local_bias -=
                    local_arr_alpha_s_in[j] * (double) local_arr_ys_in[j] *
                    svm->K(xj, xi, (int) training_data.predictors_column_number, hyper_parameters);
        }
    }

#if DEBUG_TRAIN
    std::cout << "process " << current_process << " has local bias " << local_bias << std::endl;
#endif

    MPI_Reduce(&local_bias, &svm->b, 1, MPI_DOUBLE, MPI_SUM, process_offset, MPI_COMM_WORLD);



    // reduce for master's svm properties
    MPI_Reduce(&local_arr_xs_row_size, &svm->arr_xs_row_size, 1, MPI_INT, MPI_SUM, process_offset, MPI_COMM_WORLD);
    MPI_Reduce(&local_arr_xs_in_row_size, &svm->arr_xs_in_row_size, 1, MPI_INT, MPI_SUM, process_offset, MPI_COMM_WORLD);


    if(current_process == process_offset){
        svm->b /= (double) (svm->arr_xs_row_size + svm->arr_xs_in_row_size);
    }

#if DEBUG_TRAIN
    if (current_process == process_offset) {
        std::cout << "process " << current_process << " has global bias " << svm->b << std::endl;

        print_vector(xs_sizes,available_processes);
        print_vector(xs_in_sizes, available_processes);
    }
#endif



    // send sizes to all processes for GatherV
    MPI_Allgather(&local_arr_xs_row_size,1 , MPI_INT, xs_sizes, 1, MPI_INT, MPI_COMM_WORLD);
#if DEBUG_TRAIN
    if (current_process == process_offset) {

        print_vector(xs_sizes,available_processes);

    }
#endif
    MPI_Allgather(&local_arr_xs_in_row_size,1 , MPI_INT, xs_in_sizes, 1, MPI_INT, MPI_COMM_WORLD);

#if DEBUG_TRAIN
    if (current_process == process_offset) {

        print_vector(xs_in_sizes,available_processes);

    }
#endif

    auto* array_of_sizes_x = (int *) calloc(available_processes, sizeof (int));
    auto* array_of_sizes_x_in = (int *) calloc(available_processes, sizeof (int));
    auto* array_of_displacements = (int *) calloc(available_processes, sizeof (int));
    auto* array_of_displacements_in = (int *) calloc(available_processes, sizeof (int));
    auto* array_of_displacements_alpha = (int *) calloc(available_processes, sizeof (int));
    auto* array_of_displacements_alpha_in = (int *) calloc(available_processes, sizeof (int));

    for(int l=0; l < available_processes; l++){
            array_of_sizes_x[l] = xs_sizes[l] * (int)training_data.predictors_column_number;
            array_of_sizes_x_in[l] = xs_in_sizes[l] * (int)training_data.predictors_column_number;
    }

    for(int l=1; l < available_processes; l++){
        // xs

        array_of_displacements[l] = xs_sizes[l-1] * (int)training_data.predictors_column_number + array_of_displacements[l-1];
        array_of_displacements_in[l] = xs_in_sizes[l-1] * (int)training_data.predictors_column_number + array_of_displacements_in[l-1] ;

        array_of_displacements_alpha[l] = xs_sizes[l-1] + array_of_displacements_alpha[l-1] ;
        array_of_displacements_alpha_in[l] = xs_in_sizes[l-1] + array_of_displacements_alpha_in[l-1] ;

    }

#if DEBUG_TRAIN
    if (current_process == process_offset) {

        std::cout << "Printing arrays of sizes and displacements" << std::endl;

        print_vector(array_of_sizes_x,available_processes);
        print_vector(array_of_displacements,available_processes);
        std::cout << "---------------------------------------------------" << std::endl;
        print_vector(array_of_sizes_x_in,available_processes);
        print_vector(array_of_displacements_in,available_processes);
        std::cout << "---------------------------------------------------" << std::endl;
        print_vector(array_of_displacements_alpha,available_processes);
        std::cout << "---------------------------------------------------" << std::endl;
        print_vector(array_of_displacements_alpha_in,available_processes);
        std::cout << "---------------------------------------------------" << std::endl;

    }
#endif

    svm->arr_xs_column_size = (int)training_data.predictors_column_number;
    svm->arr_xs = (double *) calloc(svm->arr_xs_row_size * training_data.predictors_column_number, sizeof(double));
    svm->arr_ys = (int *) calloc(svm->arr_xs_row_size, sizeof(int));
    svm->arr_alpha_s = (double *) calloc(svm->arr_xs_row_size, sizeof(double));
    svm->arr_xs_in = (double *) calloc(svm->arr_xs_in_row_size * training_data.predictors_column_number,sizeof(double));
    svm->arr_ys_in = (int *) calloc(svm->arr_xs_in_row_size, sizeof(int));
    svm->arr_alpha_s_in = (double *) calloc(svm->arr_xs_in_row_size, sizeof(double));


    // debug
#if DEBUG_TRAIN
    //print_matrix(local_arr_xs, local_arr_xs_row_size, training_data.predictors_column_number);
#endif
    MPI_Gatherv(local_arr_xs, local_arr_xs_row_size * (int)training_data.predictors_column_number, MPI_DOUBLE, svm->arr_xs, array_of_sizes_x, array_of_displacements, MPI_DOUBLE, process_offset, MPI_COMM_WORLD);
    MPI_Gatherv(local_arr_xs_in, local_arr_xs_in_row_size * (int)training_data.predictors_column_number, MPI_DOUBLE, svm->arr_xs_in, array_of_sizes_x_in, array_of_displacements_in, MPI_DOUBLE, process_offset, MPI_COMM_WORLD);
#if DEBUG_TRAIN
    std::cout << std::endl << std::endl;
   // if(current_process == process_offset){
   //     print_matrix(svm->arr_xs, svm->arr_xs_row_size, svm->arr_xs_column_size);
   // }

    //print_vector(local_arr_alpha_s, local_arr_xs_row_size);
#endif
    MPI_Gatherv(local_arr_alpha_s, local_arr_xs_row_size, MPI_DOUBLE, svm->arr_alpha_s, xs_sizes, array_of_displacements_alpha, MPI_DOUBLE, process_offset, MPI_COMM_WORLD);
    MPI_Gatherv(local_arr_alpha_s_in, local_arr_xs_in_row_size, MPI_DOUBLE, svm->arr_alpha_s_in, xs_in_sizes, array_of_displacements_alpha_in, MPI_DOUBLE, process_offset, MPI_COMM_WORLD);

#if DEBUG_TRAIN
    // std::cout << "Process " <<current_process << ":";
    // print_vector(local_arr_ys, local_arr_xs_row_size);

#endif
     //                                                                                                                                     same as y
    MPI_Gatherv(local_arr_ys, local_arr_xs_row_size, MPI_INT, svm->arr_ys, xs_sizes, array_of_displacements_alpha, MPI_INT, process_offset, MPI_COMM_WORLD);
    MPI_Gatherv(local_arr_ys_in, local_arr_xs_in_row_size, MPI_INT, svm->arr_ys_in, xs_in_sizes, array_of_displacements_alpha_in, MPI_INT, process_offset, MPI_COMM_WORLD);

    free(xi);
    xi = nullptr;
    free(xj);
    xj= nullptr;

    free(array_of_sizes_x);
    array_of_sizes_x = nullptr;
    free(array_of_sizes_x_in);
    array_of_sizes_x_in = nullptr;
    free(array_of_displacements);
    array_of_displacements = nullptr;
    free(array_of_displacements_in);
    array_of_displacements_in = nullptr;
    free(array_of_displacements_alpha);
    array_of_displacements_alpha = nullptr;
    free(array_of_displacements_alpha_in);
    array_of_displacements_alpha_in = nullptr;

    free(local_arr_xs);
    local_arr_xs = nullptr;
    free(local_arr_xs_in);
    local_arr_xs_in = nullptr;
    free(local_arr_ys);
    local_arr_ys = nullptr;
    free(local_arr_ys_in);
    local_arr_ys_in = nullptr;
    free(local_arr_alpha_s);
    local_arr_alpha_s = nullptr;
    free(local_arr_alpha_s_in);
    local_arr_alpha_s_in = nullptr;
    //free(y);


#if FINAL_PRINT

if(current_process == process_offset) {
    std::cout << "SVM:" << std::endl;

    std::cout << "x" << std::endl;
    print_matrix(svm->arr_xs, svm->arr_xs_row_size, svm->arr_xs_column_size);

    std::cout << "y" << std::endl;
    print_vector(svm->arr_ys, svm->arr_xs_row_size);

    std::cout << "alpha" << std::endl;
    print_vector(svm->arr_alpha_s, svm->arr_xs_row_size);

    std::cout << "x in" << std::endl;
    print_matrix(svm->arr_xs_in, svm->arr_xs_in_row_size, svm->arr_xs_column_size);

    std::cout << "y in" << std::endl;
    print_vector(svm->arr_ys_in, svm->arr_xs_in_row_size);

    std::cout << "alpha in" << std::endl;
    print_vector(svm->arr_alpha_s_in, svm->arr_xs_in_row_size);
}

#endif



    if ((svm->verbose) && (current_process == process_offset)) {
        std::cout << " bias = " << svm->b << std::endl;
        std::cout << "└───────────────────────────────────────────────────────┘\n" << std::endl;
    }


    if (svm->verbose && (current_process == process_offset)) {

        logtime();
        std::cout << "Number of support vectors *on* margin) = " << svm->arr_xs_row_size << std::endl;
        logtime();
        std::cout << "Number of support vectors *inside* margin) = " << svm->arr_xs_in_row_size << "\n" << std::endl;
    }

    if ((save_svm_flag) & (current_process == process_offset)) {

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


}

/**
 * Parallel testing function
 */

void parallel_test(Dataset test_data,
                   Kernel_SVM *svm,
                   int process_offset,
                   int available_processes,
                   int class_1 = 0) {

    // todo: implement parallel logic

    // Get the rank of the process
    int process_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);


    auto *class1_data = (double *) calloc(test_data.rows_number * test_data.predictors_column_number,
                                          sizeof(double));
    auto *class2_data = (double *) calloc(test_data.rows_number * test_data.predictors_column_number,
                                          sizeof(double));

    auto *cur_row = (double *) calloc(test_data.predictors_column_number, sizeof(double));
    int c1 = 0, c2 = 0;

    for (int i = 0; i < test_data.rows_number; i++) {

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

    int i;
    int result = 0;

    svm->correct_c1 = 0;
    for (i = 0; i < c1; i++) {
        result = (int) g_parallel(svm, class1_data + index(i, 0, test_data.predictors_column_number), process_offset, available_processes);
        if (result == -1) {
            ++svm->correct_c1;
        }
    }

    svm->correct_c2 = 0;
    for (i = 0; i < c1; i++) {
        result = (int) g_parallel(svm, class2_data + index(i, 0, test_data.predictors_column_number), process_offset, available_processes);
        if (result == 1) {
            ++svm->correct_c2;
        }
    }

    svm->accuracy =
            (double) (svm->correct_c1 + svm->correct_c2) / (double) (c1 + c2);
    svm->accuracy_c1 = (double) svm->correct_c1 / (double) c1;
    svm->accuracy_c2 = (double) svm->correct_c2 / (double) c2;

    if ((svm->verbose) && (process_rank == process_offset)) {

        std::string extended_kernel = get_extended_kernel_name(svm);

        std::cout << "\n┌───────────────── Test Results ──────────────────┐" << std::endl;
        std::cout << "  Kernel: " << extended_kernel << std::endl;
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


