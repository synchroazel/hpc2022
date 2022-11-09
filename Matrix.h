/*
class Matrix represents a matrix operator[](int i) returns a reference to the ith Row.
Row is an internal type that simply defines the operator[](int j) to return
the ith element in a Row (which is a T&)
*/

#ifndef HPC2022_MATRIX_H
#define HPC2022_MATRIX_H

#include "utility"
#include "vector"
#include "iostream"
#include "set"

#define DEBUG_MATRIX true

class Matrix {

public:
    // must be public, otherwise we cannot create mpi structs
    std::vector<double> array{};
    int m_width = 0;
    int r = 0;

    Matrix() = default;

    Matrix(int columns, int rows) : r(rows), m_width(columns), array(std::vector<double>(columns * rows)) {}

    double at(int x, int y) const { return array[index(x, y)]; }
    static double at(const std::vector<double>& matrix, int x, int y, int matrix_width){
        return matrix[y + matrix_width * x];
    }

    void modify_value(int x, int y, double new_value) { array[index(x, y)] = new_value; }
    static void modify_value(std::vector<double>& matrix, int x, int y, int matrix_width, double new_value){
        matrix[y + matrix_width * x] = new_value;
    }

    int get_rows_number() const { return this->r; }

    int get_columns_number() const { return this->m_width; }

    std::vector<double> get_row(int row_index) const {
        std::vector<double> row;
        for (int i = 0; i <= m_width; i++) { row.push_back(this->at(row_index, i)); }
        row.pop_back();
        return row;
    }

    std::vector<double> get_col(int col_index) const {
        std::vector<double> column;
        for (int i = 0; i <= r; i++) { column.push_back(this->at(i, col_index)); }
        column.pop_back();
        return column;
    }

protected:
    int index(int x, int y) const {
        return y + m_width * x;
    }

public: // presentation
    void print(bool full) const {
        std::cout << "The matrix has " << r << " rows and " << m_width << " columns" << std::endl;
        if (full) {
            for (int i = 0; i < r; i++) {
                for (int j = 0; j < m_width; j++) {
                    std::cout << at(i, j) << ", ";
                }
                std::cout << std::endl;
            }
        }
    }

    static void print(std::vector<double> x, int rows, int columns) {
        std::cout << "The matrix has " << rows << " rows and " << columns << " columns" << std::endl;

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                    std::cout << x[j + columns * i] << ", ";
            }
                std::cout << std::endl;
        }

    }
};


class Dataset {
private:
    int c = 0, r = 0;
public:

    Matrix predictor_matrix; // predictors matrix
    std::vector<int> class_vector; // class output matrix
    std::vector<int> unique_classes;
    int get_predictors_number() const { return c; }
    int get_rows_number() const { return r; }


    int class_len(int class_value) const {
        int class_size = 0;
        for (int i = 0; i < r; i++) {
            if (class_vector[i] == class_value) {
                ++class_size;
            }
        }
        return class_size;
    }


public: // constructors
    Dataset() = default;

    Dataset(int x_rows, int x_cols) {
        this->predictor_matrix = Matrix(x_cols, x_rows);
        this->class_vector = std::vector<int>(x_rows);
        this->c = x_cols + 1;
        this->r = x_rows;
        this->class_vector = {-1,1};
    }

    Dataset(const Matrix &x, const std::vector<int>& y) {
        this->predictor_matrix = x;
        this->class_vector = y;
        this->c = x.get_columns_number()+1;
        this->r = x.get_rows_number();

        // NB: this works, bug in main
        std::set<int> s( class_vector.begin(), class_vector.end() );
        this->unique_classes.assign( s.begin(), s.end() );

        // std::set<int> s;
        // unsigned size = y.size();
        // for( unsigned i = 0; i < size; ++i ) s.insert( y[i] );
        // unique_classes.assign( s.begin(), s.end() );
        // std::set<int>().swap(s);
#if DEBUG_MATRIX
        std::cout << "\nClasses are: " << std::endl;
    for (int i : this->unique_classes) {
        std::cout << i << " ";
    }
    std::cout << std::endl;
#endif
    }


public: // presentation

    void print_dataset(bool all = true) {
        std::cout << "The dataset has a total of " << get_predictors_number() << " predictors and " << get_rows_number()
                  << " rows." << std::endl;
        if (all) {
            std::cout << "Predictor matrix:" << std::endl;
            predictor_matrix.print(all);

            std::cout << "class vector:" << std::endl;
            for (auto i: class_vector)
                std::cout << i << ", ";
        }
    }

};


#endif //HPC2022_MATRIX_H
