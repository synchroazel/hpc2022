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

    int broadcaster = 0;
    char msg = 9;

    if (world_rank == broadcaster) {
        MPI_Bcast(&msg, 1, MPI_INT, broadcaster, MPI_COMM_WORLD);
        printf("Message %d broadcasted froom process rank %d.\n", msg, broadcaster);

    } else {
        printf("Message %d received from process rank %d.\n", msg, world_rank);

    }

    // Finalize the MPI environment
    MPI_Finalize();

    return 0;
}
