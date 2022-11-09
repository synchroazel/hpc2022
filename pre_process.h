//
// Created by dmmp on 06/11/22.
//

#ifndef HPC2022_PRE_PROCESS_H
#define HPC2022_PRE_PROCESS_H

#include "vector"
#include "fstream"
#include "Matrix.h"
#include "string"
#include "boost/tokenizer.hpp" // for tokenization
#include "regex"
#include "limits"

#define DEBUG_READ_DATA true
//  TODO: parallelize with MPI?
Dataset read_data_file_serial(const std::string& file_path, int rows, int columns, int target_column, char* separator,const std::string& comma_separator, bool skip_first_row=true, bool skip_first_column=true){
/*expects a scv of double*/
    // initialize
    Matrix x = Matrix(columns, rows);
#if DEBUG_READ_DATA
    x.print(false);
#endif
    std::vector<int> y = std::vector<int>(rows);
    int i=0, j; // column and row iterator
    if(skip_first_column){target_column--;}

// TODO: read
    std::string line;
    std::ifstream file (file_path);
    if (file.is_open())
    {
        if(skip_first_row){
            getline (file,line);
#if DEBUG_READ_DATA
            std::cout << "Skipping first row of the csv file" << std::endl;
#endif
        }
        while ( getline (file,line) )
        {
            j=-1;

            boost::char_separator<char> sep(separator);
            boost::tokenizer< boost::char_separator<char> > tok(line, sep);
            for(boost::tokenizer< boost::char_separator<char> >::iterator beg = tok.begin(); beg != tok.end(); ++beg)
            {
                if(skip_first_column && j == -1){
                    j++;
#if DEBUG_READ_DATA
                    std::cout << "Skipping first column of the csv file" << std::endl;
#endif
                    continue;
                } else if(!skip_first_column && j == -1){
                    j++;
                }

                const std::string& value = *beg;
                std::regex pattern ("^[0-9]"); // everything that is not a number
                std::regex_replace(value, pattern,comma_separator); // will be converted into the separator

                if(j+1 != target_column){
                    x.modify_value(i,j, std::stod(value)); // to double NB: should fix
#if DEBUG_READ_DATA
                    std::cout << "New value: " << value << " at " << i << ", " << j << std::endl;
                    x.print(true);
#endif
                } else {
                    y[i] = std::stoi(value); // to int (will be a class)
                }
                j++;
            }
            i++;
            if(i >= rows){ break;}
        }
        file.close();

    } else {
        std::cout << "Unable to open file";
        exit(1);
    }


// assign
    Dataset ris = Dataset(x,y);

// output feedback
#if DEBUG_READ_DATA
    ris.print_dataset(true);
// #else
//     ris.print(false);
#endif

    return ris;
}
// TODO: fix
void read_dataset_parallel(
        std::vector<double>& x, /*out*/
        std::vector<int>& y, /*out*/
        int x_columns, int x_rows,
        const std::string& file_path, /*in*/
        int local_rows_start, int local_columns_start, /*in*/
        int rows_to_read, int column_to_read, /*in*/
        int target_column, /*in*/
        char* separator, const std::string& comma_separator, /*in*/
        bool skip_first_row=true, bool skip_first_column=true /*in*/
        ){
    /*expects a csv of double*/
    // initialize
    if(rows_to_read == 0){rows_to_read=std::numeric_limits<int>::max();}
    if(column_to_read == 0){column_to_read=std::numeric_limits<int>::max();}

    int i=0, j; // column and row iterator
    int read_rows=0, read_columns=0;

    if(skip_first_column){target_column--;}

    std::string line;
    std::ifstream file (file_path);
    if (file.is_open())
    {
        if(skip_first_row){
            getline (file,line);
#if DEBUG_READ_DATA
            std::cout << "Skipping first row of the csv file" << std::endl;
#endif
        }
        while ( getline (file,line) )
        {
            if(i < local_rows_start){ ++i; continue;} // skip until it reaches the desired start
            j=-1;

            boost::char_separator<char> sep(separator);
            boost::tokenizer< boost::char_separator<char> > tok(line, sep);

            for(boost::tokenizer< boost::char_separator<char> >::iterator beg = tok.begin(); beg != tok.end(); ++beg)
            {
                if(skip_first_column && j == -1){
                    ++j;
#if DEBUG_READ_DATA
                    std::cout << "Skipping first column of the csv file" << std::endl;
#endif
                    continue;
                } else if(!skip_first_column && j == -1){
                    ++j;
                }

                if(j < local_columns_start){++j; continue;} // skip columns

                const std::string& value = *beg;
                // TODO: decide preprocess steps
                std::regex pattern ("^[0-9]"); // everything that is not a number
                std::regex_replace(value, pattern,comma_separator); // will be converted into the separator

                if(j+1 != target_column){
                    Matrix::modify_value(x,i,j, x_columns, std::stod(value));
#if DEBUG_READ_DATA
                    std::cout << "New value: " << value << " at " << i << ", " << j << std::endl;
                    Matrix::print(x, x_rows, x_columns);
#endif

                } else {
                    y[i] = std::stoi(value); // to int (will be a class)
                }
                ++j;
                ++read_columns;
                if(read_columns >= column_to_read){
#if DEBUG_READ_DATA
                    std::cout << "No more columns to read, next row" << std::endl;
#endif
                    break;
                }

            }
            ++i;
            ++read_rows;
            if(read_rows >= rows_to_read || i >= x_rows){
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
    Matrix::print(x, x_rows, x_columns);
#endif
}

#endif //HPC2022_PRE_PROCESS_H
