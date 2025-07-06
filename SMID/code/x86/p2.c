#include <stdio.h>
#include <stdlib.h>
#include <xmmintrin.h>

#define N 1024

double **A;
double *B;

void gaussian_elimination(double **A, double *B, int N) {
    int i, j, k;
    double temp;

    for (i = 0; i < N; i++) {
        for (j = i + 1; j < N; j++) {
            if (A[j][i] > A[i][i]) {
                for (k = 0; k < N; k++) {
                    temp = A[i][k];
                    A[i][k] = A[j][k];
                    A[j][k] = temp;
                }
                temp = B[i];
                B[i] = B[j];
                B[j] = temp;
            }
        }

        for (j = i + 1; j < N; j++) {
            temp = A[j][i] / A[i][i];
            for (k = i; k < N; k += 4) {
                __m128d row = _mm_loadu_pd(&A[j][k]);
                __m128d pivot = _mm_set1_pd(temp * A[i][k]);
                row = _mm_sub_pd(row, pivot);
                _mm_storeu_pd(&A[j][k], row);
            }
            B[j] -= temp * B[i];
        }
    }

    double X[N];
    for (i = N - 1; i >= 0; i--) {
        X[i] = B[i];
        for (j = i + 1; j < N; j++) {
            X[i] -= A[i][j] * X[j];
        }
        X[i] /= A[i][i];
    }
}

int main() {
    A = (double **)malloc(N * sizeof(double *));
    for (int i = 0; i < N; i++) {
        A[i] = (double *)malloc(N * sizeof(double));
    }
    B = (double *)malloc(N * sizeof(double));

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = rand() % 10;
        }
        B[i] = rand() % 10;
    }

    gaussian_elimination(A, B, N);

    for (int i = 0; i < N; i++) {
        free(A[i]);
    }
    free(A);
    free(B);

    return 0;
}
