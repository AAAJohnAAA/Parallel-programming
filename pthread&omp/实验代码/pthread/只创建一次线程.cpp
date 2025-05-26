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
    ele_t(*mat)[N];
    int n;
    int i;
    int begin;
    int nLines;
    pthread_mutex_t startNext;
    pthread_mutex_t finished;
} LU_data;

void* subthread_static_LU(void* _params) {
    LU_data* params = (LU_data*)_params;
    float32x4_t mat_j, mat_i, div4;

    while (1) {
        pthread_mutex_lock(&params->startNext);  // 等待主线程解锁
        int i = params->i, n = params->n;
        for (int j = params->begin; j < params->begin + params->nLines; j++) {
            if (params->mat[i][i] == 0) continue;
            ele_t div = params->mat[j][i] / params->mat[i][i];
            div4 = vmovq_n_f32(div);

            int k;
            for (k = i; k + 4 <= n; k += 4) {
                mat_j = vld1q_f32(&params->mat[j][k]);
                mat_i = vld1q_f32(&params->mat[i][k]);
                mat_j = vmlsq_f32(mat_j, div4, mat_i);
                vst1q_f32(&params->mat[j][k], mat_j);
            }
            for (; k < n; ++k)
                params->mat[j][k] -= div * params->mat[i][k];
        }
        pthread_mutex_unlock(&params->finished);  // 通知主线程已完成
    }
    return NULL;
}

void LU_static_thread(ele_t mat[N][N], int n) {
    pthread_t threads[NUM_THREADS];
    LU_data attr[NUM_THREADS];

    for (int th = 0; th < NUM_THREADS; th++) {
        attr[th].mat = mat;
        pthread_mutex_init(&attr[th].startNext, NULL);
        pthread_mutex_init(&attr[th].finished, NULL);
        pthread_mutex_lock(&attr[th].startNext);
        pthread_mutex_lock(&attr[th].finished);
        pthread_create(&threads[th], NULL, subthread_static_LU, &attr[th]);
    }

    for (int i = 0; i < n; i++) {
        int remain = n - i - 1;
        int chunk = remain / NUM_THREADS;
        int extra = remain % NUM_THREADS;
        int current = i + 1;

        for (int th = 0; th < NUM_THREADS; th++) {
            int lines = chunk + (th < extra ? 1 : 0);
            attr[th].i = i;
            attr[th].n = n;
            attr[th].begin = current;
            attr[th].nLines = lines;
            current += lines;

            if (lines > 0)
                pthread_mutex_unlock(&attr[th].startNext);  // 启动线程
        }

        for (int j = current; j < n; j++) {
            if (mat[i][i] == 0) continue;
            ele_t div = mat[j][i] / mat[i][i];
            for (int k = i; k < n; ++k)
                mat[j][k] -= div * mat[i][k];
        }

        for (int th = 0; th < NUM_THREADS; th++) {
            if (attr[th].nLines > 0)
                pthread_mutex_lock(&attr[th].finished);  // 等待线程完成
        }
    }
}

int main() {
    static ele_t mat[N][N];

    // 初始化为随机矩阵，主对角线保证非零
    srand(0);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            mat[i][j] = (i == j) ? 1000.0f : (rand() % 100) / 100.0f;

    struct timeval start, end;
    gettimeofday(&start, NULL);

    LU_static_thread(mat, N);

    gettimeofday(&end, NULL);
    double elapsed = (end.tv_sec - start.tv_sec) +
        (end.tv_usec - start.tv_usec) / 1e6;

    printf("Static-thread LU (N=%d) finished in %.6f seconds\n", N, elapsed);

    return 0;
}
