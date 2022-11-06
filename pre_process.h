//
// Created by dmmp on 06/11/22.
//

#ifndef HPC2022_PRE_PROCESS_H
#define HPC2022_PRE_PROCESS_H

#include <vector>
#include <fstream>
#include "Matrix.h"
#include "string"
#include <boost/tokenizer.hpp> // for tokenization
#include <regex>

#define DEBUG true

//  TODO: parallelize with MPI?
Dataset read_data_file(const std::string& file_path, int rows, int columns, int target_column, char* separator,const std::string& comma_separator, bool skip_first_row=true, bool skip_first_column=true){
/*expects a scv of double*/
    // initialize
    int cols = columns-1;
    int r = rows;
    if(skip_first_column)cols--;
    if(skip_first_row)r--;
    Matrix<double> x = Matrix<double>(cols, r);
#if DEBUG
    x.print(false);
#endif
    std::vector<int> y = std::vector<int>(r);
    int i=0, j; // column and row iterator
    if(skip_first_column){target_column--;}

// TODO: read
    std::string line;
    std::ifstream file (file_path);
    if (file.is_open())
    {
        if(skip_first_row){
            getline (file,line);
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
                    continue;
                }

                const std::string& value = *beg;
                std::regex pattern ("^[0-9]"); // everything that is not a number
                std::regex_replace(value, pattern,comma_separator); // will be converted into the separator

                if(j+1 != target_column){
                    x[j][i] = std::stod(value); // to double NB: should fix
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
#if DEBUG
    ris.print(true);
#else
    ris.print(false);
#endif

    return ris;
}

#endif //HPC2022_PRE_PROCESS_H
