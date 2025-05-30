#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>

#define EPS 1e-8

int main(int argc, char* argv[]) {
    int rank, size;
    const int N = 1024; // Ĭ�Ͼ����ģ�����޸Ļ�Ӳ�����ȡ��

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<std::vector<double>> A(N, std::vector<double>(N + 1));

    if (rank == 0) {
        srand(42);
        for (int i = 0; i < N; ++i)
            for (int j = 0; j <= N; ++j)
                A[i][j] = static_cast<double>(rand()) / RAND_MAX;

        std::cout << "Using MPI non-blocking communication...\n";
    }

    // �㲥�������ݣ���ʼ����
    for (int i = 0; i < N; ++i)
        MPI_Bcast(A[i].data(), N + 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double start_time = MPI_Wtime();

    for (int k = 0; k < N; ++k) {
        int owner = k % size;

        // ����Ԫ���̽�����Ԫ�У���������
        if (rank != owner) {
            MPI_Request req;
            MPI_Irecv(A[k].data(), N + 1, MPI_DOUBLE, owner, 0, MPI_COMM_WORLD, &req);
            MPI_Wait(&req, MPI_STATUS_IGNORE);
        }
        else {
            // ��Ԫ�н������������̷�����Ԫ�У���������
            for (int p = 0; p < size; ++p) {
                if (p == rank) continue;
                MPI_Request req;
                MPI_Isend(A[k].data(), N + 1, MPI_DOUBLE, p, 0, MPI_COMM_WORLD, &req);
                MPI_Wait(&req, MPI_STATUS_IGNORE); // ���� MPI_Isend+MPI_Request[] �Ż�
            }
        }

        // �����̴����为�����
        for (int i = k + 1; i < N; ++i) {
            if (i % size != rank) continue;

            double factor = A[i][k] / A[k][k];
            for (int j = k; j <= N; ++j)
                A[i][j] -= factor * A[k][j];
        }
    }

    double end_time = MPI_Wtime();

    if (rank == 0) {
        std::cout << "Finished elimination.\n";
        std::cout << "Elapsed time: " << (end_time - start_time) << " seconds\n";
    }

    MPI_Finalize();
    return 0;
}
