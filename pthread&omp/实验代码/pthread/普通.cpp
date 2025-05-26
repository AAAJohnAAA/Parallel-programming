#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N 1024
#define ele_t float

void Gauss(ele_t** A, ele_t* b, ele_t* x, int n) {
    // 分配临时矩阵和向量
    ele_t** mat = (ele_t**)malloc(n * sizeof(ele_t*));
    ele_t* vec = (ele_t*)malloc(n * sizeof(ele_t));

    for (int i = 0; i < n; i++) {
        mat[i] = (ele_t*)malloc(n * sizeof(ele_t));
        memcpy(mat[i], A[i], n * sizeof(ele_t));
    }
    memcpy(vec, b, n * sizeof(ele_t));

    // 前向消元
    for (int i = 0; i < n; i++) {
        if (mat[i][i] == 0) {
            printf("Zero pivot at row %d\n", i);
            return;
        }
        for (int j = i + 1; j < n; j++) {
            ele_t factor = mat[j][i] / mat[i][i];
            for (int k = i; k < n; k++) {
                mat[j][k] -= factor * mat[i][k];
            }
            vec[j] -= factor * vec[i];
        }
    }

    // 回代
    for (int i = n - 1; i >= 0; i--) {
        ele_t sum = 0;
        for (int j = i + 1; j < n; j++) {
            sum += mat[i][j] * x[j];
        }
        x[i] = (vec[i] - sum) / mat[i][i];
    }

    // 释放内存
    for (int i = 0; i < n; i++) free(mat[i]);
    free(mat);
    free(vec);
}

int main() {
    int n = 3;

    // 动态分配 A, b, x
    ele_t** A = (ele_t**)malloc(n * sizeof(ele_t*));
    for (int i = 0; i < n; i++) {
        A[i] = (ele_t*)malloc(n * sizeof(ele_t));
    }
    ele_t* b = (ele_t*)malloc(n * sizeof(ele_t));
    ele_t* x = (ele_t*)malloc(n * sizeof(ele_t));

    // 示例数据
    A[0][0] = 2; A[0][1] = 1; A[0][2] = -1;
    A[1][0] = -3; A[1][1] = -1; A[1][2] = 2;
    A[2][0] = -2; A[2][1] = 1; A[2][2] = 2;

    b[0] = 8;
    b[1] = -11;
    b[2] = -3;

    clock_t start = clock();
    Gauss(A, b, x, n);
    clock_t end = clock();

    printf("解向量 x:\n");
    for (int i = 0; i < n; i++) {
        printf("x[%d] = %f\n", i, x[i]);
    }

    double time_used = (double)(end - start) / CLOCKS_PER_SEC;
    printf("运行时间: %.6f 秒\n", time_used);

    // 释放内存
    for (int i = 0; i < n; i++) free(A[i]);
    free(A);
    free(b);
    free(x);

    return 0;
}
