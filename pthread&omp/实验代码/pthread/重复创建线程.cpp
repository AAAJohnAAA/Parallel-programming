#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <pthread.h>
#include <arm_neon.h>

#define N 1024
#define ele_t float
#define NUM_THREADS 4

typedef struct {
    ele_t(*mat)[N];  // 行指针
    int n;
    int i;
    int begin;
    int nLines;
} LU_data;

void* subthread_LU(void* _params) {
    LU_data* params = (LU_data*)_params;
    int i = params->i, n = params->n;
    float32x4_t mat_j, mat_i, div4;

    for (int j = params->begin; j < params->begin + params->nLines; j++) {
        if (params->mat[i][i] == 0)
            continue;
        ele_t div = params->mat[j][i] / params->mat[i][i];
        div4 = vmovq_n_f32(div);
        int k;
        for (k = i; k + 4 <= n; k += 4) {
            mat_j = vld1q_f32(&params->mat[j][k]);
            mat_i = vld1q_f32(&params->mat[i][k]);
            mat_j = vmlsq_f32(mat_j, div4, mat_i);
            vst1q_f32(&params->mat[j][k], mat_j);
        }
        for (; k < n; ++k) {
            params->mat[j][k] -= div * params->mat[i][k];
        }
    }
    return NULL;
}

void LU(ele_t mat[N][N], int n) {
    for (int i = 0; i < n; i++) {
        pthread_t threads[NUM_THREADS];
        LU_data thread_data[NUM_THREADS];

        int remain = n - i - 1;
        int chunk = remain / NUM_THREADS;
        int extra = remain % NUM_THREADS;

        int current = i + 1;

        for (int t = 0; t < NUM_THREADS; ++t) {
            int lines = chunk + (t < extra ? 1 : 0);
            thread_data[t].mat = mat;
            thread_data[t].n = n;
            thread_data[t].i = i;
            thread_data[t].begin = current;
            thread_data[t].nLines = lines;
            current += lines;

            if (lines > 0) {
                pthread_create(&threads[t], NULL, subthread_LU, &thread_data[t]);
            }
            else {
                threads[t] = 0;
            }
        }

        for (int t = 0; t < NUM_THREADS; ++t) {
            if (threads[t]) {
                pthread_join(threads[t], NULL);
            }
        }
    }
}

int main() {
    static ele_t mat[N][N];

    // 初始化矩阵为随机非奇异矩阵（主对角线设置为非零）
    srand(0);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            mat[i][j] = (i == j) ? 1000.0f : (rand() % 100) / 100.0f;

    struct timeval start, end;
    gettimeofday(&start, NULL);

    LU(mat, N);

    gettimeofday(&end, NULL);
    double time_spent = (end.tv_sec - start.tv_sec) +
        (end.tv_usec - start.tv_usec) / 1e6;

    printf("LU decomposition (N=%d) finished in %.6f seconds.\n", N, time_spent);

    return 0;
}
