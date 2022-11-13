/*
class Matrix represents a matrix operator[](int i) returns a reference to the ith Row.
Row is an internal type that simply defines the operator[](int j) to return
the ith element in a Row (which is a T&)
*/

#ifndef HPC2022_DATASET_H
#define HPC2022_DATASET_H

#include "utility"
#include "vector"
#include "iostream"
#include "set"

#define DEBUG_MATRIX true

// struct
typedef struct Dataset {
    double *predictor_matrix;
    int *class_vector;
    unsigned int predictors_column_number;
    unsigned int rows_number;
    int *unique_classes;
    unsigned int number_of_unique_classes;
} Dataset;


// access
unsigned int index(unsigned int row, unsigned int column, unsigned int column_width) {
    return row * column_width + column;
}

double get_x_element(Dataset df, unsigned int row, unsigned int column) {
    return *(df.predictor_matrix + index(row, column, df.predictors_column_number));
}


// utility functions
void get_row(Dataset df, /*in*/
             unsigned int row_index, /*in*/
             bool include_y, /*in*/
             double *output_buffer /*out*/) {
    unsigned int i = 0;

    for (; i < df.predictors_column_number; i++) {
        *(output_buffer + i) = get_x_element(df, row_index, i);
    }
    if (include_y) {
        *(output_buffer + i) = (double) (*(df.class_vector + row_index)); // put y as the last
        ++i;
    }
}

void get_column(Dataset df, /*in*/
                unsigned int column_index, /*in*/
                double *output_buffer /*out*/) {
    int i = 0;

    for (; i < df.rows_number; i++) {
        *(output_buffer + i) = get_x_element(df, i, column_index);
    }
}

void get_unique_classes(int *classes_vector, /*in*/
                        unsigned int length, /*in*/
                        unsigned int number_of_unique_classes,
                        int *output_buffer) {


    unsigned int i = 0;

    std::set<int> s;
    for (; i < length; i++) {
        s.insert(*(classes_vector + i)); // create set
    }

    std::set<int>::iterator it;
    i = 0;
    for (it = s.begin(); it != s.end(); ++it) {
        int ans = *it;
        *(output_buffer + i) = ans;
        ++i;

    }
}

int get_number_of_unique_classes(int *classes_vector, /*in*/
                                 unsigned int length /*in*/) {
    unsigned int i = 0;

    std::set<int> s;
    for (; i < length; i++) {
        s.insert(*(classes_vector + i)); // create set
    }
    return (int) s.size();

}

void modify_matrix_value(double *x, double value, unsigned int row, unsigned int column, unsigned int column_width) {
    *(x + index(row, column, column_width)) = value;
}


// presentation
template<typename T>
void print_matrix(T *x, unsigned int rows, unsigned int columns, bool metadata = false) {
    if (metadata) {
        std::cout << "The matrix has " << rows << " rows and " << columns << " columns" << std::endl;
    }

    for (unsigned int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            std::cout << x[index(i, j, columns)] << ", ";
        }
        std::cout << std::endl;
    }

}

template<typename T>
void print_vector(T *x, unsigned int rows, bool metadata = false) {
    if (metadata) {
        std::cout << "The vector has " << rows << std::endl;
    }

    for (unsigned int i = 0; i < rows; i++) {

        std::cout << x[i] << ", ";

    }
    std::cout << std::endl;
}

void print_dataset(Dataset df, bool matrix = true) {
    std::cout << "The dataset has " << df.rows_number << " rows and " << df.predictors_column_number + 1 << " columns"
              << std::endl;
    if (matrix) {
        std::cout << "Preictor matrix:" << std::endl;
        print_matrix(df.predictor_matrix, df.rows_number, df.predictors_column_number);
    }
    std::cout << "Class vector:" << std::endl;
    print_vector(df.class_vector, df.rows_number);
    std::cout << "Unique classes: (" << df.number_of_unique_classes << ")" << std::endl;
    print_vector(df.unique_classes, df.number_of_unique_classes);
}

#endif //HPC2022_DATASET_H
