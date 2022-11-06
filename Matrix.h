/*
class Matrix represents a matrix operator[](int i) returns a reference to the ith Row.
Row is an internal type that simply defines the operator[](int j) to return
the ith element in a Row (which is a T&)
*/

#ifndef HPC2022_MATRIX_H
#define HPC2022_MATRIX_H

#include <utility>
#include <vector>
#include "iostream"
// reference: https://stackoverflow.com/a/41395564
template <typename T>
class Matrix {
    // implementation
    protected:
    int col, row;

public:
    int get_columns_number(){return col;}
    int get_rows_number(){return row;}
    typedef std::vector<T> Row;

    std::vector<Row> data;

public: // interface
    Matrix(): row(0), col(0), data(0){}
    Matrix(int c, int r): row(r), col(c), data(c, std::vector<T>(r)) {}

    // allow to use matrix[i][j]
    Row & operator[](int i) {
        return data[i];
    }

public: // presentation
    void print(bool verbose=true){
        if(verbose) {
            for (int i = 0; i < col; i++) {
                for (int j = 0; j < row; j++) {
                    std::cout << data[i][j] << ", ";
                }
                std::cout << std::endl;
            }
        } else {
            std::cout << "The matrix has " << row << " rows and " << col << " columns" << std::endl;
        }
    }
};

class Dataset{
public:
    Matrix<double> predictor_matrix; // predictors matrix
    std::vector<int> class_vector; // class output matrix

    int get_predictors_number(){ return predictor_matrix.get_columns_number() + 1;}
    int get_rows_number(){ return predictor_matrix.get_rows_number();}

public: // constructor
    Dataset()= default;
    Dataset(Matrix<double> x, std::vector<int> y) {
        predictor_matrix = std::move(x);
        class_vector = std::move(y);
    }

public: // presentation
    void print(bool verbose=true){
        std::cout << "The dataset has a total of " << get_predictors_number() << "predictors and " << get_rows_number() << "rows." << std::endl;
        if(verbose){
            std::cout << "Predictor matrix:" << std::endl;
            predictor_matrix.print();

            std::cout << "class vector:" << std::endl;
            for (auto i: class_vector)
                std::cout << i << ' ';
        }
    }

};


#endif //HPC2022_MATRIX_H
