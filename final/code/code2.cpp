#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 1024  // 矩阵大小，需为 8 的倍数（64字节对齐）
#define ALIGN 64

double A[N][N] __attribute__((aligned(ALIGN)));
double b[N] __attribute__((aligned(ALIGN)));

void simd_gaussian_elimination() {
    for (int k = 0; k < N; ++k) {
        // 步骤1：主对角线元素归一化
        double pivot = A[k][k];
        __m512d pivot_vec = _mm512_set1_pd(pivot);
        for (int j = k; j < N; j += 8) {
            __m512d row_vec = _mm512_load_pd(&A[k][j]);
            row_vec = _mm512_div_pd(row_vec, pivot_vec);
            _mm512_store_pd(&A[k][j], row_vec);
        }
        b[k] /= pivot;

        // 步骤2：对k+1~N的每一行进行消元
        for (int i = k + 1; i < N; ++i) {
            double factor = A[i][k];
            __m512d factor_vec = _mm512_set1_pd(factor);

            for (int j = k; j < N; j += 8) {
                __m512d row_k = _mm512_load_pd(&A[k][j]);
                __m512d row_i = _mm512_load_pd(&A[i][j]);

                __m512d prod = _mm512_mul_pd(factor_vec, row_k);
                __m512d result = _mm512_sub_pd(row_i, prod);

                _mm512_store_pd(&A[i][j], result);
            }
            b[i] -= factor * b[k];
        }
    }

    // 步骤3：回代（标量）
    for (int i = N - 1; i >= 0; --i) {
        for (int j = i + 1; j < N; ++j)
            b[i] -= A[i][j] * b[j];
        b[i] /= A[i][i];
    }
}
