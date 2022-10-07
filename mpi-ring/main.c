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

    int max_iter = 20;
    int turn = 0;
    int loops = 0;
    int msg = 9;  //placeholder

    while (turn < max_iter) {

        int cur_turn = turn % 7;

        if (world_rank == cur_turn) {
            MPI_Send(&msg, 1, MPI_INT, cur_turn + 1, 0, MPI_COMM_WORLD);
            printf("\n%d sent msg to %d\n", world_rank, cur_turn+1);
        }

        if (world_rank == cur_turn + 1) {
            MPI_Recv(&msg, 1, MPI_INT, cur_turn, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("\n%d received msg from %d\n", cur_turn+1, cur_turn);
        }

        turn += 1;

    }

    // Finalize the MPI environment
    MPI_Finalize();

    return 0;
}
