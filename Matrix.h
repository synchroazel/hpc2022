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

class Matrix {
    std::vector<double> array{};
    int m_width = 0;
    int r = 0;
public:
    Matrix() = default;

    Matrix(int columns, int rows) : r(rows), m_width(columns), array(std::vector<double>(columns * rows)) {}

    double at(int x, int y) const { return array[index(x, y)]; }

    void modify_value(int x, int y, double new_value) { array[index(x, y)] = new_value; }

    int get_rows_number() const { return this->r; }

    int get_columns_number() const { return this->m_width; }

    std::vector<double> get_row(int row_index) const {
        std::vector<double> row;
        for (int i = 0; i < m_width; i++) { row.push_back(this->at(row_index, i)); }
        row.pop_back();
        return row;
    }

    std::vector<double> get_col(int col_index) const {
        std::vector<double> column;
        for (int i = 0; i < r; i++) { column.push_back(this->at(i, col_index)); }
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
};


class Dataset {
private:
    int c = 0, r = 0;
public:

    Matrix predictor_matrix; // predictors matrix
    std::vector<int> class_vector; // class output matrix
    int get_predictors_number() const { return c + 1; }

    int get_rows_number() const { return r; }

public: // constructors
    Dataset() = default;

    Dataset(int x_rows, int x_cols) {
        this->predictor_matrix = Matrix(x_cols, x_rows);
        this->class_vector = std::vector<int>(x_rows);
        this->c = x_cols + 1;
        this->r = x_rows;
    }

    Dataset(const Matrix &x, std::vector<int> y) {
        this->predictor_matrix = x;
        this->class_vector = std::move(y);
        this->c = x.get_columns_number();
        this->r = x.get_rows_number();
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
                std::cout << i << ' ';
        }
    }

};


#endif //HPC2022_MATRIX_H
