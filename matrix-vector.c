#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 4  // Size of the matrix and vector (NxN matrix and N vector)
// Function to initialize the matrix and vector
void initialize_matrix_and_vector(int matrix[N][N], int vector[N]) {
    for (int i = 0; i < N; i++) {
        vector[i] = i + 1;  // Vector: [1, 2, 3, ..., N]
        for (int j = 0; j < N; j++) {
            matrix[i][j] = i + j + 1;  // Simple values for matrix
        }
    }
}
int main(int argc, char** argv) {
    int world_rank, world_size;
    // Initialize the MPI environment
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int matrix[N][N];  // The matrix
    int vector[N];     // The vector
    int result[N];     // Final result vector
    int local_result[N / world_size];  // Partial result for each process
    // Only the root process initializes the matrix and vector
    if (world_rank == 0) {
        initialize_matrix_and_vector(matrix, vector);
        printf("Matrix:\n");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                printf("%d ", matrix[i][j]);
            }
            printf("\n");
        }
        printf("Vector:\n");
        for (int i = 0; i < N; i++) {
            printf("%d ", vector[i]);
        }
        printf("\n");
    }

    // Broadcast the vector to all processes
    MPI_Bcast(vector, N, MPI_INT, 0, MPI_COMM_WORLD);

    // Scatter the rows of the matrix to all processes
    int local_matrix[N / world_size][N];
    MPI_Scatter(matrix, N * (N / world_size), MPI_INT, local_matrix, N * (N / world_size), MPI_INT, 0, MPI_COMM_WORLD);

    // Each process performs its local matrix-vector multiplication
    for (int i = 0; i < N / world_size; i++) {
        local_result[i] = 0;
        for (int j = 0; j < N; j++) {
            local_result[i] += local_matrix[i][j] * vector[j];
        }
    }

    // Gather the partial results from all processes
    MPI_Gather(local_result, N / world_size, MPI_INT, result, N / world_size, MPI_INT, 0, MPI_COMM_WORLD);

    // The root process prints the final result
    if (world_rank == 0) {
        printf("Resulting vector:\n");
        for (int i = 0; i < N; i++) {
            printf("%d ", result[i]);
        }
        printf("\n");
    }

    // Finalize the MPI environment
    MPI_Finalize();
    return 0;
}
