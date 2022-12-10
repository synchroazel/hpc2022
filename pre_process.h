#ifndef HPC2022_PRE_PROCESS_H
#define HPC2022_PRE_PROCESS_H

#include "vector"
#include "fstream"
#include "Dataset.h"
#include "string"
#include "boost/tokenizer.hpp"
#include "regex"
#include "limits"

#define DEBUG_READ_DATA false
// NB: assumes pre_processed file


/**
 * Parallel reading
 */

void read_dataset_parallel(
        double *x, /*out*/
        int *y, /*out*/
        unsigned int x_columns, unsigned int x_rows,
        const std::string &file_path, /*in*/
        unsigned int local_rows_start, unsigned int local_columns_start, /*in*/
        unsigned int rows_to_read, unsigned int column_to_read, /*in*/
        unsigned int target_column, /*in*/
        char *separator /*in*/
) {
    /*expects a csv of double*/
    // initialize
    if (rows_to_read == 0) { rows_to_read = std::numeric_limits<int>::max(); }
    if (column_to_read == 0) { column_to_read = std::numeric_limits<int>::max(); }

    unsigned int i = 0, j; // column and row iterator
    unsigned int read_rows = 0, read_columns = 0;

    std::string line;
    std::ifstream file(file_path);
    if (file.is_open()) {
        while (getline(file, line)) {
            if (i < local_rows_start) {
                ++i;
                continue;
            } // skip until it reaches the desired start
            j = 0;

            boost::char_separator<char> sep(separator);
            boost::tokenizer<boost::char_separator<char> > tok(line, sep);

            for (boost::tokenizer<boost::char_separator<char> >::iterator beg = tok.begin(); beg != tok.end(); ++beg) {

                if (j < local_columns_start) {
                    ++j;
                    continue;
                } // skip columns

                const std::string &value = *beg;

                if (j + 1 != target_column) {
                    modify_matrix_value(x, std::stod(value), i, j, x_columns);
#if DEBUG_READ_DATA
                    std::cout << "New value: " << value << " at " << i << ", " << j << std::endl;
                    print_matrix(x, x_rows, x_columns);
#endif

                } else {
                    y[i] = std::stoi(value); // to int (will be a class)
#if DEBUG_READ_DATA
                    std::cout << "New value: " << value << " at " << i << " of class array" << std::endl;
                    print_vector(y, x_rows, false);
#endif
                }
                ++j;
                ++read_columns;
                if (read_columns >= column_to_read) {
#if DEBUG_READ_DATA
                    std::cout << "No more columns to read, next row" << std::endl;
#endif
                    break;
                }

            }
            ++i;
            ++read_rows;
            if (read_rows >= rows_to_read || i >= x_rows) {
#if DEBUG_READ_DATA
                std::cout << "No more rows to read, closing file." << std::endl;
#endif
                break;
            }

        }
        file.close();

    } else {
        std::cout << "Unable to open file";
        exit(1);
    }
#if DEBUG_READ_DATA
    std::cout << "Final matrix:" << std::endl;
    print_matrix(x, x_rows, x_columns);
#endif
}

#endif //HPC2022_PRE_PROCESS_H