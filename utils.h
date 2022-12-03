#ifndef HPC2022_UTILS_H
#define HPC2022_UTILS_H

#include <mpi.h>
#include <sys/time.h>
#include <cmath>

void get_current_time_formatted(char* buffer, long* millisec){
    struct tm *tm_info;
    struct timeval tv;

    gettimeofday(&tv, nullptr);


    *millisec = lrint((double) tv.tv_usec / 1000.0); // Round to nearest millisec
    if (*millisec >= 1000) { // Allow for rounding up to nearest second

        *millisec -= 1000;
        tv.tv_sec++;
    }

    tm_info = localtime(&tv.tv_sec);

    strftime(buffer, 26, "%Y-%m-%d %H:%M:%S", tm_info);
}

void logtime() {

    int process_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);

    char buffer[26];
    long millisec;

    get_current_time_formatted(buffer, &millisec);

    printf("[rank %d at %s.%03ld] ", process_rank, buffer, millisec);

}

#endif //HPC2022_UTILS_H
