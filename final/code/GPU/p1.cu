#include <iostream>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

using namespace std;
using namespace std::chrono;

__global__ void gaussian_elimination_kernel(double *matrix, double *vector, int row, int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index > row && index < size)
    {
        double factor = matrix[index * size + row] / matrix[row * size + row];
        for (int col = row + 1; col < size; ++col)
        {
            matrix[index * size + col] -= factor * matrix[row * size + col];
        }
        vector[index] -= factor * vector[row];
        matrix[index * size + row] = 0.0;
    }
}

void initialize_data(double *matrix, double *vector, int size)
{
    srand(time(0));
    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            matrix[i * size + j] = rand() % 100 + 1;
        }
        vector[i] = rand() % 100 + 1;
    }
}

void cpu_back_substitution(double *matrix, double *vector, double *solution, int size)
{
    solution[size - 1] = vector[size - 1] / matrix[size * size - 1];
    for (int i = size - 2; i >= 0; --i)
    {
        double sum = vector[i];
        for (int j = i + 1; j < size; ++j)
        {
            sum -= matrix[i * size + j] * solution[j];
        }
        solution[i] = sum / matrix[i * size + i];
    }
}

int main()
{
    int size;
    cout << "请输入矩阵的大小: ";
    cin >> size;
    double *matrix = new double[size * size];
    double *vector = new double[size];
    double *solution = new double[size];

    initialize_data(matrix, vector, size);

    double *d_matrix, *d_vector;
    cudaMalloc(&d_matrix, sizeof(double) * size * size);
    cudaMalloc(&d_vector, sizeof(double) * size);

    cudaMemcpy(d_matrix, matrix, sizeof(double) * size * size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector, vector, sizeof(double) * size, cudaMemcpyHostToDevice);

    auto start_time = high_resolution_clock::now();

    for (int row = 0; row < size - 1; ++row)
    {
        int threads_per_block = 256;
        int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;
        gaussian_elimination_kernel<<<blocks_per_grid, threads_per_block>>>(d_matrix, d_vector, row, size);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(matrix, d_matrix, sizeof(double) * size * size, cudaMemcpyDeviceToHost);
    cudaMemcpy(vector, d_vector, sizeof(double) * size, cudaMemcpyDeviceToHost);

    // 执行回代
    cpu_back_substitution(matrix, vector, solution, size);

    auto end_time = high_resolution_clock::now();
    auto duration_in_us = duration_cast<microseconds>(end_time - start_time);
    cout << "GPU 高斯消元操作完成，耗时: " << duration_in_us.count() << " 微秒" << endl;

    cudaFree(d_matrix);
    cudaFree(d_vector);
    delete[] matrix;
    delete[] vector;
    delete[] solution;

    return 0;
}
