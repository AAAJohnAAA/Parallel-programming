#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define N 1024

double **A;
double *B;

typedef struct {
    int row_start;
    int row_end;
    int pivot_row;
} thread_data_t;

void *eliminate(void *arg) {
    thread_data_t *data = (thread_data_t *)arg;
    int i, j, k;
    double temp;
    for (i = data->row_start; i < data->row_end; i++) {
        if (i > data->pivot_row) {
            temp = A[i][data->pivot_row] / A[data->pivot_row][data->pivot_row];
            for (j = data->pivot_row; j < N; j++) {
                A[i][j] -= temp * A[data->pivot_row][j];
            }
            B[i] -= temp * B[data->pivot_row];
        }
    }
    pthread_exit(NULL);
}

void gaussian_elimination(double **A, double *B, int N) {
    pthread_t threads[N];
    thread_data_t thread_data[N];
    int i, j;
    double temp;

    for (i = 0; i < N; i++) {
        for (j = i + 1; j < N; j++) {
            if (A[j][i] > A[i][i]) {
                for (int k = 0; k < N; k++) {
                    temp = A[i][k];
                    A[i][k] = A[j][k];
                    A[j][k] = temp;
                }
                temp = B[i];
                B[i] = B[j];
                B[j] = temp;
            }
        }

        for (j = 0; j < N; j++) {
            thread_data_t *data = &thread_data[j];
            data->pivot_row = i;
            data->row_start = i + 1 + j * (N - i - 1) / N;
            data->row_end = i + 1 + (j + 1) * (N - i - 1) / N;
            pthread_create(&threads[j], NULL, eliminate, (void *)data);
        }

        for (j = 0; j < N; j++) {
            pthread_join(threads[j], NULL);
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
