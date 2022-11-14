#ifndef SVM_HPP
#define SVM_HPP

#include <string>
#include <vector>
#include <functional>
#include <algorithm>
#include <boost/assign.hpp>

#include "Dataset.h"

#include "boost/archive/text_iarchive.hpp"
#include "boost/archive/text_oarchive.hpp"

// -------------------
// namespace{kernel}
// -------------------
namespace kernel {

    double linear(const std::vector<double> x1, const std::vector<double> x2, const std::vector<double> params);

    double polynomial(const std::vector<double> x1, const std::vector<double> x2, const std::vector<double> params);

    double rbf(const std::vector<double> x1, const std::vector<double> x2, const std::vector<double> params);
}

typedef std::function<double(const std::vector<double>, const std::vector<double>,
                             const std::vector<double>)> KernelFunc;


// -------------------
// class{Kernel_SVM}
// -------------------
class Kernel_SVM {
private:

    friend class boost::serialization::access;  // needed to serialize

    bool verbose;

    KernelFunc K;
    std::vector<double> params;

    // Logging function to make it easier to print stuff
    void log(const std::string str);

    template<class Archive>
    void serialize(Archive &a, const unsigned version) {
        a & arr_xs & arr_ys & arr_alpha_s & arr_xs_in & arr_ys_in & arr_alpha_s_in & b;
    }

public:

    Kernel_SVM() {};

    Kernel_SVM(const KernelFunc K_,
               const std::vector<double> params_,
               const bool verbose_) {
        this->K = K_;
        this->params = params_;
        this->verbose = verbose_;
    }

    // TODO : fix

    Kernel_SVM(
            const double    input_arr_xs[],
            const int       input_arr_ys[],
            const double    input_arr_alpha_s[],
            const double    input_arr_xs_in[],
            const int       input_arr_ys_in[],
            const double    input_arr_alpha_s_in[],
            const double    input_b) {

        /* load all deserialized parameters into arrays */

        for (int i = 0; i < sizeof(arr_ys) / sizeof(int); i++) {
            std::vector<double> temp;
            for (int j = 0; j < sizeof(arr_xs[i])/sizeof(double); j++) {
                this->arr_xs[i][j] = input_arr_xs[i * sizeof(arr_xs[i])/sizeof(double) + j];
                //temp.push_back(arr_xs[i][j]);
            }

            //this->xs.push_back(temp);

            this->arr_ys[i] = input_arr_ys[i];
            //this->ys.push_back(input_arr_ys[i]);

            this->arr_alpha_s[i] = input_arr_alpha_s[i];
            //this->alpha_s.push_back(arr_alpha_s[i]);
        }

        for (int i = 0; i < sizeof(arr_ys_in) / sizeof(int); i++) {
            std::vector<double> temp_in;
            for (int j = 0; j < sizeof(arr_xs_in[i])/sizeof(double); j++) {
                this->arr_xs_in[i][j] = input_arr_xs_in[i * sizeof(arr_xs_in[i])/sizeof(double) + j];
                //temp_in.push_back(arr_xs_in[i][j]);
            }
            //his->xs_in.push_back(temp_in);

            this->arr_ys_in[i] = input_arr_ys_in[i];
            //this->ys_in.push_back(arr_ys_in[i]);
            this->arr_alpha_s_in[i] = input_arr_alpha_s_in[i];
            //this->alpha_s_in.push_back(arr_alpha_s_in[i]);
        }

        this->b = input_b;

    }

    int tmp[90] = {0};

    std::vector<std::vector<double>> xs;
    std::vector<int> ys = {0};
    std::vector<double> alpha_s;
    std::vector<std::vector<double>> xs_in;
    std::vector<int> ys_in;
    std::vector<double> alpha_s_in;

    double arr_xs[90][90] = {0};
    int arr_ys[90] = {0};
    double arr_alpha_s[90] = {0};
    double arr_xs_in[90][90] = {0};
    int arr_ys_in[90] = {0};
    double arr_alpha_s_in[90] = {0};

    double accuracy;
    double accuracy_c1, accuracy_c2;
    size_t correct_c1, correct_c2;


    void train(Dataset training_data,
               const double C,
               const double lr,
               const double limit = 0.0001);

    void test(Dataset test_data);

    double f(const std::vector<double> x);

    double g(const std::vector<double> x);

    double b;
};


#endif