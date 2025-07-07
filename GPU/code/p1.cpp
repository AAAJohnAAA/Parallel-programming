#include <iostream>
#include <vector>

using namespace std;

void gaussianElimination(vector<vector<double>>& matrix, vector<double>& result) {
    int n = matrix.size();
    
    for (int i = 0; i < n; i++) {
        int maxRow = i;
        for (int k = i + 1; k < n; k++) {
            if (abs(matrix[k][i]) > abs(matrix[maxRow][i])) {
                maxRow = k;
            }
        }

        for (int k = i; k < n; k++) {
            swap(matrix[maxRow][k], matrix[i][k]);
        }
        swap(result[maxRow], result[i]);

        for (int k = i + 1; k < n; k++) {
            double factor = matrix[k][i] / matrix[i][i];
            for (int j = i; j < n; j++) {
                matrix[k][j] -= factor * matrix[i][j];
            }
            result[k] -= factor * result[i];
        }
    }

    for (int i = n - 1; i >= 0; i--) {
        for (int k = i - 1; k >= 0; k--) {
            double factor = matrix[k][i] / matrix[i][i];
            result[k] -= factor * result[i];
        }
        result[i] /= matrix[i][i];
    }
}

int main() {
    int n;
    cin >> n;
    
    vector<vector<double>> matrix(n, vector<double>(n));
    vector<double> result(n);
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cin >> matrix[i][j];
        }
        cin >> result[i];
    }

    gaussianElimination(matrix, result);

    for (int i = 0; i < n; i++) {
        cout << result[i] << endl;
    }

    return 0;
}
