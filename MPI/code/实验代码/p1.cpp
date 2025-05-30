#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <chrono>

const double EPS = 1e-8;

bool gaussElimination(std::vector<std::vector<double>>& A, std::vector<double>& x) {
    int n = A.size();

    for (int k = 0; k < n; ++k) {
        if (std::fabs(A[k][k]) < EPS) {
            std::cerr << "Pivot too small or zero at row " << k << "\n";
            return false;
        }

        // 消元
        for (int i = k + 1; i < n; ++i) {
            double factor = A[i][k] / A[k][k];
            for (int j = k; j <= n; ++j) {
                A[i][j] -= factor * A[k][j];
            }
        }
    }

    // 回代
    x.resize(n);
    for (int i = n - 1; i >= 0; --i) {
        x[i] = A[i][n];
        for (int j = i + 1; j < n; ++j) {
            x[i] -= A[i][j] * x[j];
        }
        x[i] /= A[i][i];
    }

    return true;
}

int main() {
    int n;
    std::cout << "Enter the number of equations: ";
    std::cin >> n;

    std::vector<std::vector<double>> A(n, std::vector<double>(n + 1));
    std::cout << "Enter the augmented matrix (row by row):\n";
    for (int i = 0; i < n; ++i)
        for (int j = 0; j <= n; ++j)
            std::cin >> A[i][j];

    std::vector<double> solution;

    // 计时开始
    auto start = std::chrono::high_resolution_clock::now();

    if (gaussElimination(A, solution)) {
        // 计时结束
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        std::cout << "Solution:\n";
        for (int i = 0; i < n; ++i)
            std::cout << "x[" << i << "] = " << std::fixed << std::setprecision(6) << solution[i] << "\n";

        std::cout << "Elapsed time: " << elapsed.count() << " seconds\n";
    }
    else {
        std::cerr << "Failed to solve the system.\n";
    }

    return 0;
}
