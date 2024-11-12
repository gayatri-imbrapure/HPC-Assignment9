#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 4  // Size of the matrices (N x N)

// Function to multiply two matrices A and B, storing the result in matrix C
void matrixMultiply(int A[N][N], int B[N][N], int C[N][N], int rowsPerProcess) {
    for (int i = 0; i < rowsPerProcess; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0;
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);  // Initialize MPI environment

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // Get the rank of the current process
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // Get the total number of processes

    int A[N][N], B[N][N], C[N][N];

    // Only the master process initializes the matrices
    if (rank == 0) {
        printf("Matrix A:\n");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A[i][j] = i + j;  // Simple initialization for A
                printf("%d ", A[i][j]);
            }
            printf("\n");
        }

        printf("\nMatrix B:\n");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                B[i][j] = i * j;  // Simple initialization for B
                printf("%d ", B[i][j]);
            }
            printf("\n");
        }
    }

    // Broadcast matrix B to all processes
    MPI_Bcast(&B, N*N, MPI_INT, 0, MPI_COMM_WORLD);

    // Number of rows per process
    int rowsPerProcess = N / size;

    int subA[rowsPerProcess][N], subC[rowsPerProcess][N];

    // Scatter rows of matrix A among processes
    MPI_Scatter(&A, rowsPerProcess*N, MPI_INT, &subA, rowsPerProcess*N, MPI_INT, 0, MPI_COMM_WORLD);

    // Perform the matrix multiplication on the subset of rows
    matrixMultiply(subA, B, subC, rowsPerProcess);

    // Gather the results from all processes
    MPI_Gather(&subC, rowsPerProcess*N, MPI_INT, &C, rowsPerProcess*N, MPI_INT, 0, MPI_COMM_WORLD);

    // The master process prints the resulting matrix C
    if (rank == 0) {
        printf("\nResulting Matrix C (A x B):\n");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                printf("%d ", C[i][j]);
            }
            printf("\n");
        }
    }

    MPI_Finalize();  // Finalize MPI environment
    return 0;
}
