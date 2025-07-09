#include <iostream>
#include <vector>
#include <cuda_runtime.h>

using namespace std;

__global__ void gaussianEliminationKernel(double* matrix, double* result, int n, int k) {
    int j = threadIdx.x + k;

    if (j < n) {
        int i = blockIdx.x;
        if (i >= k) {
            double factor = matrix[i * n + k] / matrix[k * n + k];
            matrix[i * n + j] -= factor * matrix[k * n + j];
        }
    }
}

__global__ void backSubstitutionKernel(double* matrix, double* result, int n, int k) {
    int i = threadIdx.x;

    if (i < k) {
        double factor = matrix[i * n + k] / matrix[k * n + k];
        result[i] -= factor * result[k];
    }
}

void gaussianEliminationCUDA(double* matrix, double* result, int n) {
    double* d_matrix;
    double* d_result;
    cudaMalloc(&d_matrix, n * n * sizeof(double));
    cudaMalloc(&d_result, n * sizeof(double));
    cudaMemcpy(d_matrix, matrix, n * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, result, n * sizeof(double), cudaMemcpyHostToDevice);

    for (int k = 0; k < n; k++) {
        int gridSize = n;
        int blockSize = n - k;

        gaussianEliminationKernel<<<gridSize, blockSize>>>(d_matrix, d_result, n, k);
        cudaDeviceSynchronize();
    }

    for (int k = n - 1; k >= 0; k--) {
        int blockSize = n - k;
        backSubstitutionKernel<<<1, blockSize>>>(d_matrix, d_result, n, k);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(result, d_result, n * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_matrix);
    cudaFree(d_result);
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

    vector<double> flat_matrix(n * n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            flat_matrix[i * n + j] = matrix[i][j];
        }
    }

    gaussianEliminationCUDA(flat_matrix.data(), result.data(), n);

    for (int i = 0; i < n; i++) {
        cout << result[i] << endl;
    }

    return 0;
}
