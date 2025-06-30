#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 1024

double A[N][N];
double b[N];
double x[N];

void gaussian_elimination_openmp() {
    int i, j, k;

    for (k = 0; k < N; ++k) {
        // 步骤1：主对角线归一化（串行）
        double pivot = A[k][k];
        for (j = k; j < N; ++j)
            A[k][j] /= pivot;
        b[k] /= pivot;

        // 步骤2：对第k行以下的所有行进行并行消元
        #pragma omp parallel for private(i, j) shared(A, b, k)
        for (i = k + 1; i < N; ++i) {
            double factor = A[i][k];
            for (j = k; j < N; ++j)
                A[i][j] -= factor * A[k][j];
            b[i] -= factor * b[k];
        }
    }

    // 步骤3：回代求解（串行）
    for (i = N - 1; i >= 0; --i) {
        x[i] = b[i];
        for (j = i + 1; j < N; ++j)
            x[i] -= A[i][j] * x[j];
    }
}
