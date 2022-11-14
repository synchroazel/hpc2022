#include <iostream>
#include <string>
#include <vector>
#include <cmath>

#include "svm.hpp"
#include "fstream"

#include "Dataset.h"


#define WRITE_SV_TO_CSV true
#define DEBUG_SUPPORT_VECTORS false


// ---------------------------------------
// namespace{kernel} -> function{linear}
// ---------------------------------------
double kernel::linear(const std::vector<double> x1,
                      const std::vector<double> x2,
                      const std::vector<double> params) {

    size_t i;
    double ans;

    if (x1.size() != x2.size()) {
        std::cerr << "Error : Couldn't match the number of elements for inner product." << std::endl;
        std::exit(-1);
    }

    ans = 0.0;
    for (i = 0; i < x1.size(); i++) {
        ans += x1[i] * x2[i];
    }

    return ans;

}


// -------------------------------------------
// namespace{kernel} -> function{polynomial}
// -------------------------------------------
double
kernel::polynomial(const std::vector<double> x1,
                   const std::vector<double> x2,
                   const std::vector<double> params) {

    size_t i;
    double ans;

    if (x1.size() != x2.size()) {
        std::cerr << "Error : Couldn't match the number of elements for inner product." << std::endl;
        std::exit(-1);
    } else if (params.size() != 2) {
        std::cerr << "Error : Couldn't match the number of hyper-parameters." << std::endl;
        std::exit(-1);
    }

    ans = 0.0;
    for (i = 0; i < x1.size(); i++) {
        ans += x1[i] * x2[i];
    }
    ans += params[0];
    ans = std::pow(ans, params[1]);

    return ans;

}


// ------------------------------------
// namespace{kernel} -> function{rbf}
// ------------------------------------
double kernel::rbf(const std::vector<double> x1,
                   const std::vector<double> x2,
                   const std::vector<double> params) {

    size_t i;
    double ans;

    if (x1.size() != x2.size()) {
        std::cerr << "Error : Couldn't match the number of elements for inner product." << std::endl;
        std::exit(-1);
    } else if (params.size() != 1) {
        std::cerr << "Error : Couldn't match the number of hyper-parameters." << std::endl;
        std::exit(-1);
    }

    ans = 0.0;
    for (i = 0; i < x1.size(); i++) {
        ans += (x1[i] - x2[i]) * (x1[i] - x2[i]);
    }
    ans = std::exp(-params[0] * ans);

    return ans;

}


// ------------------------------------
// class{Kernel_SVM} -> function{log}
// ------------------------------------
void Kernel_SVM::log(const std::string str) {
    if (this->verbose) {
        std::cout << str << std::flush;
    }
    return;
}


// --------------------------------------
// class{Kernel_SVM} -> function{train}
// --------------------------------------
void Kernel_SVM::train(Dataset training_data,
                       const double C,
                       const double lr,
                       const double limit) {


    // split all training data into class1 and class2 data
    std::vector<std::vector<double>> class1_data;
    std::vector<std::vector<double>> class2_data;

    for (size_t i = 0; i < training_data.rows_number; i++) {

        auto *cur_row = (double *) calloc(training_data.predictors_column_number, sizeof(double));
        get_row(training_data, i, false, cur_row);

        //convert cur_row to vector of doubles
        std::vector<double> cur_row_vector;
        for (size_t j = 0; j < training_data.predictors_column_number; j++) {
            cur_row_vector.push_back(cur_row[j]);
        }

        if (training_data.class_vector[i] == 0) {
            class1_data.push_back(cur_row_vector);
        } else if (training_data.class_vector[i] == 1) {
            class2_data.push_back(cur_row_vector);
        }
    }

    constexpr double eps = 0.0000001;

    size_t i, j;
    size_t N, Ns, Ns_in;
    bool judge;
    double item1, item2, item3;
    double delta;
    double beta;
    double error;
    std::vector<std::vector<double>> x;
    std::vector<int> y;
    // std::vector<double> alpha;

    // (1.1) Set class 1 data
    for (i = 0; i < class1_data.size(); i++) {
        x.push_back(class1_data[i]);
        y.push_back(1);
    }

    // (1.2) Set class 2 data
    for (i = 0; i < class2_data.size(); i++) {
        x.push_back(class2_data[i]);
        y.push_back(-1);
    }

    // (2) Set Lagrange Multiplier and Parameters
    N = x.size();

    double alpha[N];

    beta = 1.0;

    // (3) Training
    this->log("\n");
    this->log("/////////////////////// Training ///////////////////////\n");
    do {

        judge = false;
        error = 0.0;

        // (3.1) Update Alpha
        for (i = 0; i < N; i++) {

            // Compute the partial derivative with respect to alpha

            item1 = 0.0;
            for (j = 0; j < N; j++) {
                item1 += alpha[j] * (double) y[i] * (double) y[j] * this->K(x[i], x[j], this->params);
            }

            // Set item 2
            item2 = 0.0;
            for (j = 0; j < N; j++) {
                item2 += alpha[j] * (double) y[i] * (double) y[j];
            }

            // Set such partial derivative to Delta

            delta = 1.0 - item1 - beta * item2;

            // Update
            alpha[i] += lr * delta;
            if (alpha[i] < 0.0) {
                alpha[i] = 0.0;
            } else if (alpha[i] > C) {
                alpha[i] = C;
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
        this->log("\rerror: " + std::to_string(error));

    } while (judge);
    this->log("\n");
    this->log("////////////////////////////////////////////////////////\n");

    // (4.1) Description for support vectors
    Ns = 0;
    Ns_in = 0;
    this->xs = std::vector<std::vector<double>>();
    this->ys = std::vector<int>();
    this->alpha_s = std::vector<double>();
    this->xs_in = std::vector<std::vector<double>>();
    this->ys_in = std::vector<int>();
    this->alpha_s_in = std::vector<double>();


    for (i = 0; i < N; i++) {
        if ((eps < alpha[i]) && (alpha[i] < C - eps)) {
            this->xs.push_back(x[i]);
            this->ys.push_back(y[i]);
            this->alpha_s.push_back(alpha[i]);


            for (size_t j = 0; j < training_data.predictors_column_number; j++) {
                this->arr_xs[Ns][j] = x[i][j];
            }

            this->arr_ys[Ns] = y[i];
            this->arr_alpha_s[Ns] = alpha[i];

            Ns++;

        } else if (alpha[i] >= C - eps) {
            this->xs_in.push_back(x[i]);
            this->ys_in.push_back(y[i]);
            this->alpha_s_in.push_back(alpha[i]);


            for (size_t j = 0; j < training_data.predictors_column_number; j++) {
                this->arr_xs_in[Ns_in][j] = x[i][j];
            }

            this->arr_ys_in[Ns_in] = y[i];
            this->arr_alpha_s_in[Ns_in] = alpha[i];

            Ns_in++;
        }

    }



#if DEBUG_SUPPORT_VECTORS

    //print this->xs
    this->log("xs:\n");
    for (i = 0; i < this->xs.size(); i++) {
        for (j = 0; j < this->xs[i].size(); j++) {
            this->log(std::to_string(this->xs[i][j]) + " ");
        }
        this->log("\n");
    }

    //print this->ys
    this->log("ys: ");
    for (i = 0; i < this->ys.size(); i++) {
        this->log(std::to_string(this->arr_ys[i]) + " ");
    } this->log("\n");

    //print this->alpha_s
    this->log("alpha_s: ");
    for (i = 0; i < this->alpha_s.size(); i++) {
        this->log(std::to_string(this->arr_alpha_s[i]) + " ");
    } this->log("\n");

    //print this->xs_in
    this->log("xs_in:\n");
    for (i = 0; i < this->xs_in.size(); i++) {
        for (j = 0; j < this->xs_in[i].size(); j++) {
            this->log(std::to_string(this->xs_in[i][j]) + " ");
        }
        this->log(" | ");
    }

    //print this->ys_in
    this->log("ys_in: ");
    for (i = 0; i < this->ys_in.size(); i++) {
        this->log(std::to_string(this->arr_ys_in[i]) + " ");
    } this->log("\n");

    //print this->alpha_s_in
    this->log("alpha_s_in: ");
    for (i = 0; i < this->alpha_s_in.size(); i++) {
        this->log(std::to_string(this->arr_alpha_s_in[i]) + " ");
    } this->log("\n");

#endif

    this->log("Ns (number of support vectors on margin) = " + std::to_string(Ns) + "\n");

    this->log("Ns_in (number of support vectors inside margin) = " + std::to_string(Ns_in) + "\n");


#if WRITE_SV_TO_CSV

    /** Write all support vectors to file **/

    std::ofstream myfile;

    // If not exist, create directory to save the model params

    std::string dir = "../saved_svm";
    if (!std::__fs::filesystem::exists(dir)) {
        std::__fs::filesystem::create_directory(dir);
    }

    myfile.open("../saved_svm/sv_on.csv");

    /* Support vectors on margin */
    /* x1, x2, ..., xn, y, alpha */

    for (i = 0; i < Ns; i++) {
        for (j = 0; j < this->xs[i].size(); j++) {
            myfile << std::to_string(this->xs[i][j]) + ", ";
        }
        myfile << std::to_string(this->ys[i]) + ", " + std::to_string(this->arr_alpha_s[i]) + "\n";
    }

    this->log("Support vectors on margin saved to sv_on.csv.\n");
    myfile.close();

    myfile.open("../saved_svm/sv_in.csv");

    /* Support vectors inside margin */
    /* x1, x2, ..., xn, y, alpha */

    for (i = 0; i < Ns_in; i++) {
        for (j = 0; j < this->xs_in[i].size(); j++) {
            myfile << std::to_string(this->xs_in[i][j]) + ", ";
        }
        myfile << std::to_string(this->arr_ys_in[i]) + ", " + std::to_string(this->arr_alpha_s_in[i]) + "\n";
    }

    this->log("Support vectors inside margin saved to sv_in.csv.\n");
    myfile.close();

#endif

    // Update the bias
    this->b = 0.0;
    for (i = 0; i < Ns; i++) {
        this->b += (double) this->ys[i];
        for (j = 0; j < Ns; j++) {
            this->b -=
                    this->arr_alpha_s[j] * (double) this->arr_ys[j] * this->K(this->xs[j], this->xs[i], this->params);
        }
        for (j = 0; j < Ns_in; j++) {
            this->b -=
                    this->arr_alpha_s_in[j] * (double) this->arr_ys_in[j] *
                    this->K(this->xs_in[j], this->xs[i], this->params);
        }
    }
    this->b /= (double) Ns;
    this->log("bias = " + std::to_string(this->b) + "\n");
    this->log("////////////////////////////////////////////////////////\n\n");

    return;
}


// -------------------------------------
// class{Kernel_SVM} -> function{test}
// -------------------------------------
void Kernel_SVM::test(Dataset test_data) {


    // split all training data into class1 and class2 data

    std::vector<std::vector<double>> class1_data;
    std::vector<std::vector<double>> class2_data;

    for (size_t i = 0; i < test_data.rows_number; i++) {

        auto *cur_row = (double *) calloc(test_data.predictors_column_number, sizeof(double));
        get_row(test_data, i, false, cur_row);

        //convert cur_row to vector of doubles
        std::vector<double> cur_row_vector;
        for (size_t j = 0; j < test_data.predictors_column_number; j++) {
            cur_row_vector.push_back(cur_row[j]);
        }

        if (test_data.class_vector[i] == 0) {
            class1_data.push_back(cur_row_vector);
        } else if (test_data.class_vector[i] == 1) {
            class2_data.push_back(cur_row_vector);
        }
    }

    size_t i;

    this->correct_c1 = 0;
    for (i = 0; i < class1_data.size(); i++) {
        if (this->g(class1_data[i]) == 1) {
            this->correct_c1++;
        }
    }

    this->correct_c2 = 0;
    for (i = 0; i < class2_data.size(); i++) {
        if (this->g(class2_data[i]) == -1) {
            this->correct_c2++;
        }
    }

    this->accuracy =
            (double) (this->correct_c1 + this->correct_c2) / (double) (class1_data.size() + class2_data.size());
    this->accuracy_c1 = (double) this->correct_c1 / (double) class1_data.size();
    this->accuracy_c2 = (double) this->correct_c2 / (double) class2_data.size();

    return;
}


// ----------------------------------
// class{Kernel_SVM} -> function{f}
// ----------------------------------
double Kernel_SVM::f(const std::vector<double> x) {

    size_t i;
    double ans;

    ans = 0.0;
    for (i = 0; i < this->xs.size(); i++) {
        ans += this->alpha_s[i] * this->ys[i] * this->K(this->xs[i], x, this->params);
    }
    for (i = 0; i < this->xs_in.size(); i++) {
        ans += this->alpha_s_in[i] * this->ys_in[i] * this->K(this->xs_in[i], x, this->params);
    }
    ans += this->b;

    return ans;
}


// ----------------------------------
// class{Kernel_SVM} -> function{g}
// ----------------------------------
double Kernel_SVM::g(const std::vector<double> x) {

    double fx;
    int gx;

    fx = this->f(x);
    if (fx >= 0.0) {
        gx = 1;
    } else {
        gx = -1;
    }

    return gx;
}


void Set_Kernel(std::string ker_type,
                KernelFunc &K,
                std::vector<double> &params) {

    if (ker_type == "linear") {
        K = kernel::linear;
    } else if (ker_type == "polynomial") {
        K = kernel::polynomial;
        // params = {c, d};
    } else if (ker_type == "rbf") {
        K = kernel::rbf;
        // params = {gamma};
    }
}