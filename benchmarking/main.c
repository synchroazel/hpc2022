#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

#define SEC_CONVERSION 1000
#define MB_CONVERSION 1048576

int main() {

    // Initialize the MPI environment
    MPI_Init(NULL, NULL);

    // Find out rank, size
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    double t1, t2;
    int a = 0;
    int b = 1;
    int count;
    int size;
    int iters = 1000000; // 1e-6

    if (world_rank == 0) {
        FILE *fptr = fopen("data.tsv", "w");
        printf("n\ttime (sec)\trate (MB/sec)\n");
        fprintf(fptr, "n\ttime (sec)\trate (MB/sec)\n");
        fclose(fptr);
    }

    FILE *fptr = fopen("data.tsv", "a");

    for (int ex = 0; ex <= 20; ex++) {

        count = 0;

        size = (int) pow(2, ex);

        char *message = (char *) calloc(size, sizeof(char));

        t1 = MPI_Wtime();

        while (count < iters * 2) {

            if (world_rank == a) {
                if (count % 2 == 0) {
                    MPI_Send(message, size, MPI_CHAR, b, count, MPI_COMM_WORLD);
                    //printf("message was sent by process %d to process %d.\n", a, b);
                } else {
                    MPI_Recv(message, size, MPI_CHAR, b, count, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }

            } else if (world_rank == b) {
                if (count % 2 == 1) {
                    MPI_Send(message, size, MPI_CHAR, a, count, MPI_COMM_WORLD);
                    //printf("message was sent by process %d to process %d.\n", b, a);
                } else {
                    MPI_Recv(message, size, MPI_CHAR, a, count, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }

            count += 1;

        }

        t2 = MPI_Wtime();

        double tot_time = t2 - t1;
        double trip_time = tot_time / iters;

        free(message);

        if (world_rank == a) {

            printf("%d\t%.10f\t%.10f\n",
                   size,
                   trip_time * SEC_CONVERSION,
                   (double) size / MB_CONVERSION / (trip_time * SEC_CONVERSION)
            );


            fprintf(fptr, "%d\t%.10f\t%.10f\n", size, trip_time * SEC_CONVERSION,
                    (double) size / MB_CONVERSION / (trip_time * SEC_CONVERSION)
            );
        }

    }

    fclose(fptr);

    // Finalize the MPI environment
    MPI_Finalize();

    return 0;

}