 <stdio.h>
 <mpi.h>

int main() {

    // Initialize the MPI environment
    MPI_Init(NULL, NULL);

    // Find out rank, size
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    printf("Hello world from process rank %d of %d.\n", world_rank, world_size);

    // Finalize the MPI environment
    MPI_Finalize();

    return 0;
}
