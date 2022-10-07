#include <stdio.h>
#include <mpi.h>

int main() {

    // Initialize the MPI environment
    MPI_Init(NULL, NULL);

    // Find out rank, size
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int num = 5;   // !!ASK!! same for each process!
    int red_num;
    int root = 0;  // rank of the process to reduce on

    MPI_Reduce(&num, &red_num, 1, MPI_INT, MPI_SUM, root, MPI_COMM_WORLD);

    if (world_rank == root) {
        printf("Hi, I'm process rank %d and `num` is %d.\n", root, red_num);
    }

    // Finalize the MPI environment
    MPI_Finalize();

    return 0;
}
