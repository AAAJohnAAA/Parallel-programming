#include <iostream>
#include <chrono>

using namespace std;
using namespace std::chrono;

#define SIZE 5500

int matrix[SIZE][SIZE];
int b[SIZE];
int sum[SIZE];

void init() {
    for (int i = 0; i < SIZE; i++) {
        b[i] = i;
        for (int j = 0; j < SIZE; j++) {
            matrix[i][j] = i + j;
        }
    }
}

void p1() {
    for (int i = 0; i < SIZE; i++) {
        sum[i] = 0;
        for (int j = 0; j < SIZE; j++) {
            sum[i] += matrix[j][i] * b[j];
        }
    }
}

void p2() {
    for (int i = 0; i < SIZE; i++) sum[i] = 0;

    for (int j = 0; j < SIZE; j++) {
        for (int i = 0; i < SIZE; i++) {
            sum[i] += matrix[j][i] * b[j];
        }
    }
}

int main() {
    auto start = high_resolution_clock::now();
    init();
    p1();
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    cout << "col_major time: " << duration.count() << " ms" << endl;

    start = high_resolution_clock::now();
    init();
    p2();
    stop = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(stop - start);
    cout << "row_major time: " << duration.count() << " ms" << endl;

    return 0;
}
