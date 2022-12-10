#include <iostream>
#include <getopt.h>
#include "regex.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

void generate_dummy_script() {
    FILE *file_to_write;
    file_to_write = fopen("./dummy_shell_script.sh", "w");
    std::string current_string = "#!/bin/bash\n"
                                 "\n"
                                 "#PBS -l select=4:ncpus=16:mem=4gb\n"
                                 "#PBS -l walltime=0:60:00\n"
                                 "#PBS -q short_cpuQ\n"
                                 "\n"
                                 "module load mpich-3.2\n"
                                 "\n"
                                 "export PERFORMANCE_CHECKS=TRUE\n"
                                 "\n"
                                 "mpirun.actual -np 64 ~/hpc2022/cmake-build-debug/hpc2022 -l tuning \\\n"
                                 "                                   -i ~/hpc2022/data/iris_train.csv \\\n"
                                 "                                   -I ~/hpc2022/data/iris_validation.csv \\\n"
                                 "                                   -H ~/hpc2022/hyperparameters.json \\\n"
                                 "                                   -r 70 -R 30 -c 5 -t 5";
    fwrite(current_string.c_str(), sizeof(char), current_string.length(), file_to_write);
    fclose(file_to_write);
    std::cout << "Shell script was saved as: './dummy_shell_script'" << std::endl;
    exit(0);
}

void read_rows_columns(const std::string &filepath, std::string &tr_rows, std::string &val_rows, std::string &tst_rows,
                       std::string &cols) {

    FILE *file_to_read;
    file_to_read = fopen(filepath.c_str(), "rb");

    //string to be searched
    char *line;
    size_t len = 0;
    ssize_t read;

    // source: https://gist.github.com/ianmackinnon/3294587

    // regex expression for pattern to be searched
    regex_t regex;

    size_t maxMatches = 2;
    size_t maxGroups = 3;
    const char *pattern = "([0-9]+)";

    regmatch_t groupArray[maxGroups];

    unsigned int m;
    char *cursor;

    if (regcomp(&regex, pattern, REG_EXTENDED)) {
        printf("Could not compile regular expression.\n");
        exit(1);
    }


    for (int i = 0; i < 3; i++) {
        read = getline(&line, &len, file_to_read);
        m = 0;
        cursor = line;

        for (m = 0; m < maxMatches; m++) {
            if (regexec(&regex, cursor, maxGroups, groupArray, 0))
                break;  // No more matches

            unsigned int g = 0;
            unsigned int offset = 0;
            for (g = 0; g < maxGroups; g++) {
                if (groupArray[g].rm_so == (size_t) -1)
                    break;  // No more groups

                if (g == 0)
                    offset = groupArray[g].rm_eo;

                char cursorCopy[strlen(cursor) + 1];
                strcpy(cursorCopy, cursor);
                cursorCopy[groupArray[g].rm_eo] = 0;

                if (i == 0 && m == 0) {
                    tr_rows = cursorCopy + groupArray[g].rm_so;
                } else if (i == 0 && m == 1) {
                    cols = cursorCopy + groupArray[g].rm_so;
                } else if (i == 1 && m == 0) {
                    val_rows = cursorCopy + groupArray[g].rm_so;
                } else if (i == 2 && m == 0) {
                    tst_rows = cursorCopy + groupArray[g].rm_so;
                }

                // printf("Match %u, Group %u: [%2u-%2u]: %s\n",
                //        m, g, groupArray[g].rm_so, groupArray[g].rm_eo,
                //        cursorCopy + groupArray[g].rm_so);
            }
            cursor += offset;
        }
    }

    regfree(&regex);
    free(line);
    line = nullptr;
    fclose(file_to_read);

}

void print_usage(const std::string &program_name) {
    std::cout << "Usage:  " << program_name << " options\n"
              << "  -h  --help                   Display this usage information.\n"
              << "  -D  --dummy                  Creates a dummy shell script\n"
              << "  -P  --cpus                   MPI processes to run\n"
              << "  -W  --walltime               Walltime (es: '1:00:00')\n"
              << "  -Q  --queue                  queue (es: 'short_cpuQ')\n"
              << "  -m  --memory                 memory to allocate on the cluster\n"
              << "  -l  --logic                  Program logic, may be 'training', 'testing' or 'tuning'.\n"
              << "  -p  --parallel-tuning        Set tuning logic. Can be 'split' or 'sequential'. Default adapts to dataset size.\n"
              << "  -i  --path1                  First input path supplied, may be interpreted as training path or testing path.\n"
              << "  -I  --path2                  Second input path supplied, in tuning logic is interpreted as validation.\n"
              << "  -t  --target-column          Index of the target column.\n"
              << "  -r  --row-col-filepath       Path to file containing rows and columns (generated by ds_preprocessing.py).\n"
              << "  -H  --hyperparameters-path   Path to the hyperparameters file.\n"
              << "  -s  --svm-path               Path to the SVM file.\n"
              << "  -S  --save-dir-path          Folder path for saving SVM files.\n"
              << "  -M  --tuning-table-dir-path  Folder path for saving tuning table files.\n"
              << "  -k  --kernel                 Kernel type, may be 'l' (linear), 'p' (polynomial), 'r' (rbf) or 's' (sigmoid).\n"
              << "  -C  --cost                   Cost parameter.\n"
              << "  -g  --gamma                  Gamma parameter.\n"
              << "  -O  --coef0                  Coef0 parameter.\n"
              << "  -d  --degree                 Degree parameter.\n"
              << "  -T  --learning_rate          Learning rate parameter.\n"
              << "  -E  --eps                    Epsilon parameter.\n"
              << "  -L  --limit                  Limit parameter.\n"
              << "  -v  --verbose                Print verbose messages.\n"
              << std::endl;
    exit(0);
}

int main(int argc, char *argv[]) {

    std::string filepath_svm;

    std::string save_svm_dir_path;

    std::string logic;
    std::string parallel_tuning;

    std::string target_column;
    std::string columns;
    std::string row1;
    std::string row2;
    std::string row3;
    std::string hyperparameters_path;
    std::string tuning_table_dir_path;
    std::string kernel;
    std::string cost;
    std::string gamma;
    std::string coef0;
    std::string degree;
    std::string learning_rate;
    std::string eps;
    std::string limit;
    bool verbose = false;
    bool optimized = false;

    std::string row_col_filepath;
    std::string ncpus;
    std::string memory;
    std::string walltime;
    std::string queue;

    // ----------------------- deal with cli args --------------------------------------------
    int next_option = 0;

    /* A string listing valid short options letters. */
    const char *const short_options = "hDP:m:l:p:i:I:t:r:H:s:S:M:k:C:g:O:d:T:E:L:vW:Q:o";

    /* An array describing valid long options.  */
    const struct option long_options[] = {
            {"help",                  0, nullptr, 'h'},
            {"dummy",                 0, nullptr, 'D'},
            {"cpus",                  1, nullptr, 'P'},
            {"memory",                2, nullptr, 'm'},
            {"walltime",              2, nullptr, 'W'},
            {"queue",                 2, nullptr, 'Q'},
            {"optimized",             0, nullptr, 'o'},
            {"logic",                 1, nullptr, 'l'},
            {"parallel-tuning",       2, nullptr, 'p'},
            {"path1",                 1, nullptr, 'i'},
            {"path2",                 1, nullptr, 'I'},
            {"target-column",         1, nullptr, 't'},
            {"row-col-filepath",      1, nullptr, 'r'},
            {"hyperparameters-path",  2, nullptr, 'H'},
            {"svm-path",              2, nullptr, 's'},
            {"save-dir-path",         2, nullptr, 'S'},
            {"tuning-table-dir-path", 2, nullptr, 'M'},
            {"kernel",                2, nullptr, 'k'},
            {"cost",                  2, nullptr, 'C'},
            {"gamma",                 2, nullptr, 'g'},
            {"coef0",                 2, nullptr, 'O'},
            {"degree",                2, nullptr, 'd'},
            {"learning_rate",         2, nullptr, 'T'},
            {"eps",                   2, nullptr, 'E'},
            {"limit",                 1, nullptr, 'L'},
            {"verbose",               0, nullptr, 'v'},
            {nullptr,                 0, nullptr, 0}
    };

    /**
     * Parameters initialization
     */

    std::string p1;
    std::string p2;

    std::string program_name = argv[0];
    std::string program_path;


    /**
     * CLI arguments parsing
     */


    // ---------- assignments -----------
    do {

        next_option = getopt_long(argc, argv, short_options, long_options, nullptr);

        switch (next_option) {

            case 'h': {
                /* -h or --help */
                print_usage(program_name);
                exit(0);
            }
            case 'o': {
                /* -o or --optimized */
                optimized = true;
                break;
            }
            case 'D': {
                /* -D or --dummy */
                generate_dummy_script();
                exit(0);
            }
            case 'P': {
                /* -P or --cpus */
                ncpus = optarg;
                break;
            }
            case 'W': {
                /* -W or --walltime */
                walltime = optarg;
                break;
            }
            case 'Q': {
                /* -Q or --queue */
                queue = optarg;
                break;
            }
            case 'm': {
                /* -m or --memory */
                memory = optarg;
                break;
            }
            case 'l': {
                /* -l or --logic */
                logic = optarg;
                break;
            }
            case 'p': {   /* -p or --parallel-tuning */
                parallel_tuning = optarg;
                break;
            }
            case 'i': {  /* -i or --path1 */
                p1 = optarg;
                break;
            }
            case 'I': {
                p2 = optarg;
                break;
            }
            case 't': {
                /* -t or --target_column */
                target_column = optarg;
                break;
            }
            case 'r': {
                row_col_filepath = optarg;
                break;
            }
            case 'H': {   /* -H or --hparameters_path */
                hyperparameters_path = optarg;
                break;
            }
            case 's': {  /* -s or --svm-path */

                filepath_svm = optarg;
                break;
            }

            case 'S': {  /* -S or --save-dir-path */
                save_svm_dir_path = optarg;
                break;
            }
            case 'M': {  /* -M or --tuning-table-dir-path */
                tuning_table_dir_path = optarg;
                break;
            }

            case 'k': {  /* -k or --kernel */
                kernel = optarg;
                break;
            }

            case 'C': { /* -C or --cost */
                cost = optarg;
                break;
            }

            case 'g': { /* -g or --gamma */
                gamma = optarg;
                break;
            }

            case 'O': { /* -O or --coef0 */
                coef0 = optarg;
                break;
            }

            case 'd': { /* -d or --degree */
                degree = optarg;
                break;
            }
            case 'T': {   /* -T or --learning_rate */
                learning_rate = optarg;
                break;
            }

            case 'E': {  /* -E or --eps */
                eps = optarg;
                break;
            }

            case 'L': {  /* -L or --limit */
                limit = optarg;
                break;
            }

            case 'v': { /* -v or --verbose */
                verbose = true;
                break;
            }

            case '?': {  /* The user specified an invalid option */

                // print_usage(1);
            } // ?

            case -1: {   /* Done with options */
                break;
            }

            default: {
                /* Something else unexpected */
                exit(1);
            }

        }

    } while (next_option != -1);

    if (row_col_filepath.empty() || p1.empty() || logic.empty() || target_column.empty()) {
        std::cout << "Arguments missing!" << std::endl;
        exit(1);
    }

    // file creation
    FILE *file_to_write;
    /* --------------name -------------------------------*/
    std::string name = "submit_" + ncpus + "_processes.sh";
    /*-----------------------------------------------------*/
    std::string tmpstr;
    const char *filename = name.c_str();

    int nodes, cpus, total_mpi_processes = atoi(ncpus.c_str());
    int tries[] = {8, 7, 6, 5, 4, 3, 2, 1};

    // decide split of nodes and cpus
    for (int trie: tries) {
        if (total_mpi_processes % trie == 0) {
            cpus = trie;
            break;
        }
    }
    nodes = total_mpi_processes / cpus;

    file_to_write = fopen(filename, "w");

    fputs("#!/bin/bash\n", file_to_write);

    if (!memory.empty()) {
        memory = ":mem=" + memory + "gb";
    } else {
        memory = ":mem=4gb";
    }
    tmpstr = std::string("#PBS -l select=" + std::to_string(nodes) + ":ncpus=" + std::to_string(cpus) + memory) + "\n";
    fputs(tmpstr.c_str(), file_to_write);
    tmpstr.clear();

    if (!walltime.empty()) {
        tmpstr = "#PBS -l walltime=" + walltime + "\n";
    } else {
        tmpstr = "#PBS -l walltime=0:60:00\n";
    }
    fputs(tmpstr.c_str(), file_to_write);
    tmpstr.clear();

    if (!queue.empty()) {
        tmpstr = "#PBS -q " + queue + "\n";
    } else {
        tmpstr = "#PBS -q short_cpuQ\n";
    }
    fputs(tmpstr.c_str(), file_to_write);
    tmpstr.clear();

    fputs("module load mpich-3.2\n", file_to_write);
    fputs("export PERFORMANCE_CHECKS=TRUE\n", file_to_write);

    fputs("\n\n\n\n", file_to_write);

    /* create strings for shell script */

    std::string begin = "mpirun.actual -np " + std::to_string(total_mpi_processes);
    std::string tabs = "\t\t\t\t";
    std::string eol = " \\\n";

    // program path
    if (optimized) {
        program_path = " ~/hpc2022/cmake-build-debug/hpc2022 ";
    } else {
        program_path = " ~/hpc2022/cmake-build-optimized/hpc2022 ";
    }
    read_rows_columns(row_col_filepath, row1, row2, row3, columns);
    if (logic == "training") {
        row1 = tabs + "-r " + row1 + " ";
        row2.clear();
    } else if (logic == "tuning") {
        row1 = tabs + "-r " + row1 + " ";
        row2 = tabs + "-R " + row2 + " ";
    } else {
        row1 = tabs + "-r " + row3 + " ";
        row2.clear();
    }
    columns = "-c " + columns + " ";
    target_column = "-t " + target_column + eol;


    // logic
    logic = "-l " + logic + eol;
    // p1
    p1 = tabs + "-i " + p1 + eol;
    // p2
    if (!p2.empty()) {
        p2 = tabs + "-I " + p2 + eol;
    }
    // hyperparameters path
    if (!hyperparameters_path.empty()) {
        hyperparameters_path = tabs + "-H " + hyperparameters_path + eol;
    }
    // parallel-tuning
    if (!parallel_tuning.empty()) {
        parallel_tuning = tabs + "-p " + parallel_tuning + eol;
    }
    // svm path
    if (!filepath_svm.empty()) {
        filepath_svm = tabs + "-s " + filepath_svm + eol;
    }
    // svm save dire path
    if (!save_svm_dir_path.empty()) {
        save_svm_dir_path = tabs + "-S " + save_svm_dir_path + eol;
    }

    // tuning table dir path
    if (!tuning_table_dir_path.empty()) {
        tuning_table_dir_path = tabs + "-M " + tuning_table_dir_path + eol;
    }

    // kernel
    if (!kernel.empty()) {
        kernel = " -k " + kernel;
    }
    // cost
    if (!cost.empty()) {
        cost = " -C " + cost;
    }
    // gamma
    if (!gamma.empty()) {
        gamma = " -g " + gamma;
    }
    // intercept
    if (!coef0.empty()) {
        coef0 = " -O " + coef0;
    }
    // degree
    if (!degree.empty()) {
        degree = " -d " + degree;
    }

    // -T learning rate
    if (!learning_rate.empty()) {
        learning_rate = " -T " + learning_rate;
    }

    // -E epsilon
    if (!eps.empty()) {
        eps = " -E " + eps;
    }
    // -L limit
    if (!limit.empty()) {
        limit = " -L " + limit;
    }

    std::string verb;
    if (verbose) {
        verb = " -v";
    }

    // final string
    tmpstr = begin + program_path + logic +
             p1 +
             p2 +
             hyperparameters_path +
             row1 + row2 + columns + target_column +
             parallel_tuning +
             filepath_svm +
             save_svm_dir_path +
             tuning_table_dir_path +
             tabs + kernel + cost + gamma + coef0 + degree + learning_rate + eps + limit + verb + "             ";
    fputs(tmpstr.c_str(), file_to_write);
    fclose(file_to_write);

}