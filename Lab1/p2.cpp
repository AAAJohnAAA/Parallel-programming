#include <iostream>
#include <ctime>

using namespace std;

#define SIZE 90000000

double* a;
double sum = 0;

// 初始化数据
void init() {
    a = new double[SIZE];
    for (int i = 0; i < SIZE; i++) a[i] = i;
    sum = 0;
}

// 释放内存
void destroy() {
    delete[] a;
    sum = 0;
}

// 平凡算法
void p1() {
    for (int i = 0; i < SIZE; i++) sum += a[i];
}

// 超标量优化算法
void p2() {
    double sum1 = 0, sum2 = 0;
    for (int i = 0; i < SIZE - 1; i += 2) {
        sum1 += a[i];
        sum2 += a[i + 1];
    }
    if (SIZE % 2 != 0) sum1 += a[SIZE - 1];
    sum = sum1 + sum2;
}

// 递归算法
void p3(int n) {
    if (n <= 1) return;

    int m = n / 2;
    for (int i = 0; i < m; i++)
        a[i] += a[n - i - 1];

    p3(m);  // 递归调用
}

int main() {
    clock_t start, stop;
    double durationTime = 0.0;

    // 平凡算法
    init();
    start = clock();
    p1();
    stop = clock();
    destroy();
    durationTime = (double)(stop - start) / CLOCKS_PER_SEC * 1000.0;
    cout << "p1 (trivial algorithm) time: " << durationTime << " ms" << endl;

    // 超标量优化算法
    init();
    start = clock();
    p2();
    stop = clock();
    destroy();
    durationTime = (double)(stop - start) / CLOCKS_PER_SEC * 1000.0;
    cout << "p2 (superscalar optimization) time: " << durationTime << " ms" << endl;

    // 递归算法
    init();
    start = clock();
    p3(SIZE);
    stop = clock();
    destroy();
    durationTime = (double)(stop - start) / CLOCKS_PER_SEC * 1000.0;
    cout << "p3 (recursive algorithm) time: " << durationTime << " ms" << endl;

    return 0;
}