#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <vector>
#include <cmath>

#define EPS 1e-8

int main(int argc, char* argv[]) {
    int rank, size;
    const int N = 1024; // 可更改为命令行参数或动态输入

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 每个进程存一整行的矩阵
    std::vector<std::vector<double>> A(N, std::vector<double>(N + 1));

    if (rank == 0) {
        srand(42);
        for (int i = 0; i < N; ++i)
            for (int j = 0; j <= N; ++j)
                A[i][j] = static_cast<double>(rand()) / RAND_MAX;

        std::cout << "Starting parallel Gaussian elimination...\n";
    }

    // 广播整个矩阵
    for (int i = 0; i < N; ++i)
        MPI_Bcast(A[i].data(), N + 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double t_start = MPI_Wtime();

    for (int k = 0; k < N; ++k) {
        int owner = k % size;

        // 所有进程广播当前主元行
        MPI_Bcast(A[k].data(), N + 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);

        // 每个进程处理它负责的行
#pragma omp parallel for
        for (int i = k + 1; i < N; ++i) {
            if (i % size != rank) continue;
            double factor = A[i][k] / A[k][k];

#pragma omp simd
            for (int j = k; j <= N; ++j) {
                A[i][j] -= factor * A[k][j];
            }
        }
    }

    double t_end = MPI_Wtime();

    if (rank == 0) {
        std::cout << "Parallel elimination completed.\n";
        std::cout << "Elapsed time: " << (t_end - t_start) << " seconds\n";
    }

    MPI_Finalize();
    return 0;
}
