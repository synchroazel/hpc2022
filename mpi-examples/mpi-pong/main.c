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

    int player_a = 0;  // starting player
    int player_b = 1;
    int ping_pong = 0;
    char ball = 'o';

    while (ping_pong < 10) {

        if (ping_pong % 2 == 0) {

            if (world_rank == player_a) {
                MPI_Send(&ball, 1, MPI_CHAR, player_b, ping_pong, MPI_COMM_WORLD);

            } else if (world_rank == player_b) {
                MPI_Recv(&ball, 1, MPI_CHAR, player_a, ping_pong, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                printf("PING! ");
                printf("Ball was sent by process %d to process %d.\n", player_a, player_b);

            }

        } else {

            if (world_rank == player_b) {
                MPI_Send(&ball, 1, MPI_CHAR, player_a, ping_pong, MPI_COMM_WORLD);

            } else if (world_rank == player_a) {
                MPI_Recv(&ball, 1, MPI_CHAR, player_b, ping_pong, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                printf("\nPONG! ");
                printf("Ball was sent by process %d to process %d.\n", player_b, player_a);

            }


        }
            ping_pong = ping_pong + 1;
            printf("Ping pong count: %d.\n\n", ping_pong);
    }

    // Finalize the MPI environment
    MPI_Finalize();

    return 0;

}