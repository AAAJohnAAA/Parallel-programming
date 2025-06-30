#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>

// 高斯消去：解线性方程组 Ax = b，结果写入 x
bool gaussian_elimination(std::vector<std::vector<double>> A, std::vector<double> b, std::vector<double>& x) {
    const double EPS = 1e-10;
    int n = A.size();
    x.resize(n);

    // 消元过程
    for (int k = 0; k < n; ++k) {
        // 选主元（可选，提升稳定性）
        int maxRow = k;
        for (int i = k + 1; i < n; ++i) {
            if (std::fabs(A[i][k]) > std::fabs(A[maxRow][k]))
                maxRow = i;
        }
        if (std::fabs(A[maxRow][k]) < EPS) {
            std::cerr << "矩阵奇异，无法求解。\n";
            return false;
        }
        std::swap(A[k], A[maxRow]);
        std::swap(b[k], b[maxRow]);

        // 消元
        for (int i = k + 1; i < n; ++i) {
            double factor = A[i][k] / A[k][k];
            for (int j = k; j < n; ++j) {
                A[i][j] -= factor * A[k][j];
            }
            b[i] -= factor * b[k];
        }
    }

    // 回代
    for (int i = n - 1; i >= 0; --i) {
        double sum = b[i];
        for (int j = i + 1; j < n; ++j) {
            sum -= A[i][j] * x[j];
        }
        x[i] = sum / A[i][i];
    }

    return true;
}

// 测试示例
int main() {
    std::vector<std::vector<double>> A = {
        {2, 1, -1},
        {-3, -1, 2},
        {-2, 1, 2}
    };
    std::vector<double> b = {8, -11, -3};
    std::vector<double> x;

    if (gaussian_elimination(A, b, x)) {
        std::cout << "解为：\n";
        for (size_t i = 0; i < x.size(); ++i) {
            std::cout << "x[" << i << "] = " << std::fixed << std::setprecision(6) << x[i] << '\n';
        }
    }

    return 0;
}
