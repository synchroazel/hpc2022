#include <iostream>
#include <string>
#include <cmath>
#include "svm.h"
#include "Dataset.h"

#define DEBUG_SUPPORT_VECTORS false


// ---------------------------------------
// namespace{kernel} -> function{linear}
// ---------------------------------------
double kernel::linear(const double* x1,
                      const double* x2,
                      size_t size,
                      double params[4] = nullptr) { // cost, gamma, coef0, degree, in this case none

    size_t i;
    double ans= 0.0;

    // u*v
    for (i = 0; i < size; i++) {
        //        x1   *    x2
        ans += x1[i] * x2[i];
    }

    return ans;
}


// -------------------------------------------
// namespace{kernel} -> function{polynomial}
// -------------------------------------------
double kernel::polynomial(const double* x1,
                          const double* x2,
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


// ------------------------------------
// namespace{kernel} -> function{radial}
// ------------------------------------
double kernel::radial(const double* x1,
                   const double* x2,
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

// ------------------------------------
// namespace{kernel} -> function{sigmoid}
// ------------------------------------
double kernel::sigmoid(const double* x1,
                      const double* x2,
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



// --------------------------------------
// class{Kernel_SVM} -> function{train}
// --------------------------------------

// training data must have only two classes
void train(const Dataset& training_data,
                       Kernel_SVM* svm,
                       double hyper_parameters[4],
                       const double lr,
                       const double limit,
           bool save_svm_flag = true,
           size_t class_1=0,
           const double eps = 0.0000001
                       ) {


    int y[training_data.rows_number];  // array
    memset(y, 0, training_data.rows_number * sizeof (int ));
    size_t i=0;
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

    i=0; // i will be useful later
    size_t j;

    bool judge;
    double item1, item2, item3;
    double delta;
    double beta;
    double error;

    // (2) Set Lagrange Multiplier and Parameters
    double alpha[N];
    memset(alpha, 0, N * sizeof (double ));

    beta = 1.0;

    // tmp rows
    double xi[training_data.predictors_column_number];
    double xj[training_data.predictors_column_number];

    // (3) Training
    if(svm->verbose){
        log("\n");
        log("/////////////////////// Training ///////////////////////\n");
    }

    do {

        judge = false;
        error = 0.0;

        // (3.1) Update Alpha
        for (i = 0; i < N; i++) {

            // Compute the partial derivative with respect to alpha

            item1 = 0;
            for (j = 0; j < N; j++) {
                get_row(training_data,i, false, xi);
                get_row(training_data,j, false, xj);
                item1 += alpha[j] * (double) y[i] * (double) y[j] * svm->K(xi, xj, training_data.predictors_column_number, hyper_parameters);
            }

            // Set item 2
            item2 = 0;
            for (j = 0; j < N; j++) {
                item2 += alpha[j] * (double) y[i] * (double) y[j];
            }

            // Set such partial derivative to Delta

            delta = 1.0 - item1 - beta * item2;

            // Update
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

        // (3.2) Update bias Beta
        item3 = 0.0;
        for (i = 0; i < N; i++) {
            item3 += alpha[i] * (double) y[i];
        }
        beta += item3 * item3 / 2.0;

        // (3.3) Output Residual Error
        if(svm->verbose){
            log("\rerror: " + std::to_string(error));
        }

    } while (judge);

    if(svm->verbose){
        log("\n");
        log("////////////////////////////////////////////////////////\n");
    }


    // (4.1) Description for support vectors


    // initialize, then realloc
    // NB: N are the rows of the new matrix
    // need to check for memory leaks, we will try another approach in the meanwhile
    svm->arr_xs = (double *) calloc(N * training_data.predictors_column_number, sizeof (double )); // matrix
    svm->arr_ys = (int *) calloc(N, sizeof (int ));
    svm->arr_alpha_s = (double *) calloc(N, sizeof (double ));


    svm->arr_xs_row_size = 0;
    svm->arr_xs_column_size = training_data.predictors_column_number;
    svm->arr_alpha_size = 0;

    svm->arr_xs_in = (double *) calloc(N * training_data.predictors_column_number, sizeof (double )); // matrix
    svm->arr_ys_in = (int *) calloc(N, sizeof (int ));
    svm->arr_alpha_s_in = (double *) calloc(N, sizeof (double ));



    svm->arr_xs_in_row_size = 0;
    svm->arr_xs_in_column_size = training_data.predictors_column_number;
    svm->arr_alpha_in_size = 0;

    int sv =0, svi = 0;
    for (i = 0; i < N; i++) {
        if ((eps < alpha[i]) && (alpha[i] < hyper_parameters[0]/*cost*/ - eps)) {
            // support vectors outside the margin
            get_row(training_data,i, false, xi);
            memcpy(svm->arr_xs + index(sv, 0, training_data.predictors_column_number),xi,training_data.predictors_column_number * sizeof (double ));
            ++svm->arr_xs_row_size;

            *(svm->arr_ys + sv) = y[i];

            *(svm->arr_alpha_s + sv) = alpha[i];
            ++svm->arr_alpha_size;

            ++sv;

        } else if (alpha[i] >= hyper_parameters[0]/*cost*/ - eps) {
            // support vectors inside the margin
            get_row(training_data,i, false, xi);
            memcpy(svm->arr_xs_in + index(svi, 0, training_data.predictors_column_number),xi,training_data.predictors_column_number * sizeof (double ));
            ++svm->arr_xs_in_row_size;

            *(svm->arr_ys_in + svi) = y[i];

            *(svm->arr_alpha_s_in + svi) = alpha[i];
            ++svm->arr_alpha_in_size;

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
    svm->arr_xs = (double *) reallocarray(svm->arr_xs, svm->arr_xs_row_size * svm->arr_xs_column_size, sizeof (double ));
    svm->arr_ys = (int *) reallocarray(svm->arr_ys, svm->arr_xs_row_size, sizeof (int));
    svm->arr_alpha_s = (double *) reallocarray(svm->arr_alpha_s, svm->arr_alpha_size, sizeof (double ));

    svm->arr_xs_in = (double *) reallocarray(svm->arr_xs_in, svm->arr_xs_in_row_size * svm->arr_xs_in_column_size, sizeof (double ));
    svm->arr_ys_in = (int *) reallocarray(svm->arr_ys_in, svm->arr_xs_in_row_size, sizeof (int));
    svm->arr_alpha_s_in = (double *) reallocarray(svm->arr_alpha_s_in, svm->arr_alpha_in_size, sizeof (double ));
//
    // Update the bias
    svm->b = 0.0;
    for (i = 0; i < svm->arr_xs_row_size; i++) {
        svm->b += (double) svm->arr_ys[i];
        for (j = 0; j < svm->arr_xs_row_size; j++) {
            get_row(svm->arr_xs,i, false, xi);
            get_row(svm->arr_xs,j, false, xj);
            svm->b -=
                    svm->arr_alpha_s[j] * (double) svm->arr_ys[j] * svm->K(xj, xi, svm->arr_xs_column_size, svm->params);
        }
        for (j = 0; j < svm->arr_xs_in_row_size; j++) {
            get_row(svm->arr_xs_in,i, false, xi);
            get_row(svm->arr_xs_in,j, false, xj);
            svm->b -=
                    svm->arr_alpha_s_in[j] * (double) svm->arr_ys_in[j] *
                    svm->K(xj, xi, svm->arr_xs_column_size, svm->params);
        }
    }
    svm->b /= (double) (svm->arr_xs_row_size + svm->arr_xs_in_row_size);
    log("bias = " + std::to_string(svm->b) + "\n");
    log("////////////////////////////////////////////////////////\n\n");


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

    if(svm->verbose){
        log("Ns (number of support vectors on margin) = " + std::to_string(svm->arr_xs_row_size) + "\n");

        log("Ns_in (number of support vectors inside margin) = " + std::to_string(svm->arr_xs_in_row_size) + "\n");
    }





    if(save_svm_flag){
        /** Write svm to file **/
        std::string s;
        // If not exist, create directory to save the model params
        switch (svm->kernel_type) {
            case 'l':{
                s = "../saved_svm/linear_C" + std::to_string(hyper_parameters[0]) + ".dat";
                break;
            }
            case 'r':{
                s = "../saved_svm/radial" + std::string (1, svm->kernel_type) +
                    "_C" + std::to_string(hyper_parameters[0])+
                    "_G" + std::to_string(hyper_parameters[1])+
                    ".dat";
                break;
            }
            case 's':{
                s = "../saved_svm/sigmoid" + std::string (1, svm->kernel_type) +
                    "_C" + std::to_string(hyper_parameters[0])+
                    "_G" + std::to_string(hyper_parameters[1])+
                    "_O" + std::to_string(hyper_parameters[2])+
                    ".dat";
                break;
            }
            case 'p':{
                s = "../saved_svm/polynomial" + std::string (1, svm->kernel_type) +
                    "_C" + std::to_string(hyper_parameters[0])+
                    "_G" + std::to_string(hyper_parameters[1])+
                    "_O" + std::to_string(hyper_parameters[2])+
                    "_D" + std::to_string(hyper_parameters[3])+
                    ".dat";
                break;
            }
        }


        save_svm(svm, s);
    }



    // TODO: capire di cosa fare il free


}


// -------------------------------------
// class{Kernel_SVM} -> function{test}
// -------------------------------------
void test(Dataset test_data,
          Kernel_SVM* svm,
          size_t class_1=0) {


    // split all training data into class1 and class2 data

    auto *class1_data = (double *) calloc(test_data.rows_number * test_data.predictors_column_number, sizeof (double ));
    auto *class2_data = (double *) calloc(test_data.rows_number * test_data.predictors_column_number, sizeof (double ));

    auto *cur_row = (double *) calloc(test_data.predictors_column_number, sizeof(double));
    size_t c1=0, c2=0;

    for (size_t i = 0; i < test_data.rows_number; i++) {

        get_row(test_data, i, false, cur_row);

        if (test_data.class_vector[i] == test_data.unique_classes[class_1]) {
            memcpy(class1_data + (c1 * test_data.predictors_column_number),  cur_row,test_data.predictors_column_number * sizeof (double ));
            ++c1;
        } else{
            memcpy(class2_data + (c2 * test_data.predictors_column_number),  cur_row,test_data.predictors_column_number * sizeof (double ));
            ++c2;
        }
    }

    size_t i;

    svm->correct_c1 = 0;
    for (i = 0; i < c1; i++) {
        if (g(svm, class1_data + index(i, 0, test_data.predictors_column_number)) == 1) {
            ++svm->correct_c1;
        }
    }

    svm->correct_c2 = 0;
    for (i = 0; i < c1; i++) {
        if (g(svm, class2_data + index(i, 0, test_data.predictors_column_number)) == -1) {
            ++svm->correct_c2;
        }
    }


    svm->accuracy =
            (double) (svm->correct_c1 + svm->correct_c2) / (double) (c1 + c2);
    svm->accuracy_c1 = (double) svm->correct_c1 / (double) c1;
    svm->accuracy_c2 = (double) svm->correct_c2 / (double) c2;

    free(class1_data);
    class1_data = nullptr;
    free(class2_data);
    class2_data= nullptr;
    free(cur_row);
    cur_row = nullptr;

}


