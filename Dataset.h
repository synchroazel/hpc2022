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


/**
 * Struct for Dataset
 */

typedef struct Dataset {
    double *predictor_matrix;
    int *class_vector;
    unsigned int predictors_column_number;
    unsigned int rows_number;
    int *unique_classes;
    unsigned int number_of_unique_classes;
} Dataset;


/**
 * Accessing the dataset
 */

unsigned int index(unsigned int row, unsigned int column, unsigned int column_width) {
    return row * column_width + column;
}

double get_x_element(const Dataset &df, unsigned int row, unsigned int column) {
    return *(df.predictor_matrix + index(row, column, df.predictors_column_number));
}


/**
 * Utility functions (for matrix)
 */

void get_row(const double *x, /*in*/
             unsigned int row_index, /*in*/
             unsigned int column_width, /*in*/
             double *output_buffer) {

    for (int j = 0; j < column_width; j++) {
        *(output_buffer + j) = *(x + index(row_index, j, column_width));
    }
}

void get_column(const double *x, /*in*/
                unsigned int column_index, /*in*/
                unsigned int column_width, /*in*/
                unsigned int row_width,
                double *output_buffer /*out*/) {

    for (int i = 0; i < row_width; i++) {
        *(output_buffer + i) = *(x + index(i, column_index, column_width));
    }
}


/**
 * Utility functions (for Dataset)
 */

void get_row(const Dataset &df, /*in*/
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

void get_column(const Dataset &df, /*in*/
                unsigned int column_index, /*in*/
                double *output_buffer /*out*/) {
    int i = 0;

    for (; i < df.rows_number; i++) {
        *(output_buffer + i) = get_x_element(df, i, column_index);
    }
}

void get_unique_classes(int *classes_vector, /*in*/
                        unsigned int length, /*in*/
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

void set_row_values(double *x, const double *row_values, unsigned int row, unsigned int column_width) {
    // TODO: search for a better approach

    memcpy(x + index(row, 0, column_width), row_values, column_width);
    // for(int i=0; i < column_width; i++){
    //     modify_matrix_value(x, *(row_values + i), row, i, column_width);
    // }
}


/**
 * Presentation
 */

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

void print_dataset(const Dataset &df, bool matrix = true) {
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
