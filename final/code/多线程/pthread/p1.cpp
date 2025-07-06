#include <stdio.h>
#include <stdlib.h>

#define N 1024  
void gaussian_elimination(double **A, double *B) {
    int i, j, k;
    double temp;

    for (i = 0; i < N; i++) {
        for (k = i + 1; k < N; k++) {
            if (A[i][i] == 0) {
                printf("无法进行消元，主元为零！\n");
                return;
            }
            if (A[k][i] > A[i][i]) {
                for (j = 0; j < N; j++) {
                    temp = A[i][j];
                    A[i][j] = A[k][j];
                    A[k][j] = temp;
                }
                temp = B[i];
                B[i] = B[k];
                B[k] = temp;
            }
        }
        for (k = i + 1; k < N; k++) {
            temp = A[k][i] / A[i][i];
            for (j = i; j < N; j++) {
                A[k][j] -= temp * A[i][j];
            }
            B[k] -= temp * B[i];
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
    double **A = (double **)malloc(N * sizeof(double *));
    for (int i = 0; i < N; i++) {
        A[i] = (double *)malloc(N * sizeof(double));
    }
    double *B = (double *)malloc(N * sizeof(double));

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = rand() % 10;  
        }
        B[i] = rand() % 10;  
    }

    gaussian_elimination(A, B);
    for (int i = 0; i < N; i++) {
        free(A[i]);
    }
    free(A);
    free(B);

    return 0;
}
