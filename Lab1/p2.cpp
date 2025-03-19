#include <iostream>
#include <ctime>

using namespace std;

#define SIZE 90000000

double* a;
double sum = 0;

// ��ʼ������
void init() {
    a = new double[SIZE];
    for (int i = 0; i < SIZE; i++) a[i] = i;
    sum = 0;
}

// �ͷ��ڴ�
void destroy() {
    delete[] a;
    sum = 0;
}

// ƽ���㷨
void p1() {
    for (int i = 0; i < SIZE; i++) sum += a[i];
}

// �������Ż��㷨
void p2() {
    double sum1 = 0, sum2 = 0;
    for (int i = 0; i < SIZE - 1; i += 2) {
        sum1 += a[i];
        sum2 += a[i + 1];
    }
    if (SIZE % 2 != 0) sum1 += a[SIZE - 1];
    sum = sum1 + sum2;
}

// �ݹ��㷨
void p3(int n) {
    if (n <= 1) return;

    int m = n / 2;
    for (int i = 0; i < m; i++)
        a[i] += a[n - i - 1];

    p3(m);  // �ݹ����
}

int main() {
    clock_t start, stop;
    double durationTime = 0.0;

    // ƽ���㷨
    init();
    start = clock();
    p1();
    stop = clock();
    destroy();
    durationTime = (double)(stop - start) / CLOCKS_PER_SEC * 1000.0;
    cout << "p1 (trivial algorithm) time: " << durationTime << " ms" << endl;

    // �������Ż��㷨
    init();
    start = clock();
    p2();
    stop = clock();
    destroy();
    durationTime = (double)(stop - start) / CLOCKS_PER_SEC * 1000.0;
    cout << "p2 (superscalar optimization) time: " << durationTime << " ms" << endl;

    // �ݹ��㷨
    init();
    start = clock();
    p3(SIZE);
    stop = clock();
    destroy();
    durationTime = (double)(stop - start) / CLOCKS_PER_SEC * 1000.0;
    cout << "p3 (recursive algorithm) time: " << durationTime << " ms" << endl;

    return 0;
}