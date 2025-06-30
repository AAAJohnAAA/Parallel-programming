#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

#define N 1024  // 矩阵维度，可替换为任意大小

void print_matrix(vector<vector<double>>& A, vector<double>& b, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) cout << A[i][j] << " ";
        cout << "| " << b[i] << endl;
    }
}

int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // 当前进程编号
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // 总进程数

    vector<vector<double>> A(N, vector<double>(N));
    vector<double> b(N);
    vector<double> x(N);

    // 初始化矩阵（只在根进程做）
    if (rank == 0) {
        srand(42);
        for (int i = 0; i < N; i++) {
            b[i] = rand() % 100;
            for (int j = 0; j < N; j++) {
                A[i][j] = (i == j) ? 1.0 + rand() % 5 : rand() % 10;
            }
        }
    }

    // 每个进程本地的数据大小（行数）
    int local_rows = N / size;
    vector<vector<double>> local_A(local_rows, vector<double>(N));
    vector<double> local_b(local_rows);

    // 拆分矩阵行给各进程
    for (int i = 0; i < N; i++) {
        if (rank == 0) {
            for (int p = 0; p < size; p++) {
                int offset = p * local_rows;
                if (p == 0) {
                    for (int j = 0; j < local_rows; j++) {
                        local_A[j] = A[j];
                        local_b[j] = b[j];
                    }
                } else {
                    for (int j = 0; j < local_rows; j++) {
                        MPI_Send(A[offset + j].data(), N, MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
                        MPI_Send(&b[offset + j], 1, MPI_DOUBLE, p, 1, MPI_COMM_WORLD);
                    }
                }
            }
        } else {
            for (int j = 0; j < local_rows; j++) {
                MPI_Recv(local_A[j].data(), N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&local_b[j], 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }

    // 并行高斯消元
    for (int k = 0; k < N; k++) {
        int owner = k / local_rows;
        vector<double> pivot_row(N);
        double pivot_b;

        if (rank == owner) {
            int local_k = k % local_rows;
            pivot_row = local_A[local_k];
            pivot_b = local_b[local_k];
        }

        MPI_Bcast(pivot_row.data(), N, MPI_DOUBLE, owner, MPI_COMM_WORLD);
        MPI_Bcast(&pivot_b, 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);

        for (int i = 0; i < local_rows; i++) {
            int global_i = i + rank * local_rows;
            if (global_i > k) {
                double factor = local_A[i][k] / pivot_row[k];
                for (int j = k; j < N; j++) {
                    local_A[i][j] -= factor * pivot_row[j];
                }
                local_b[i] -= factor * pivot_b;
            }
        }
    }

    // 回代（集中回代，结果聚合到根进程）
    MPI_Gather(local_b.data(), local_rows, MPI_DOUBLE, b.data(), local_rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    for (int i = 0; i < N; i++) {
        if (rank == 0) MPI_Gather(local_A[i].data(), N, MPI_DOUBLE, A[i].data(), N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        for (int i = N - 1; i >= 0; i--) {
            x[i] = b[i];
            for (int j = i + 1; j < N; j++) {
                x[i] -= A[i][j] * x[j];
            }
            x[i] /= A[i][i];
        }

        cout << "[MPI] 解向量 x[0..9] 示例：" << endl;
        for (int i = 0; i < min(N, 10); i++) cout << x[i] << " ";
        cout << endl;
    }

    MPI_Finalize();
    return 0;
}
