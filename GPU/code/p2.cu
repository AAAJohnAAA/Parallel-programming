#include <iostream>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
using namespace std;

__global__ void gaussian_kernel(double *matrix, double *vector, int index, int size)
{
    int rowIdx = index + 1 + blockIdx.x;
    int colIdx = index + 1 + threadIdx.x;

    if (rowIdx < size && colIdx < size)
    {
        double factorA = matrix[rowIdx * size + index];
        double pivot = matrix[index * size + index];
        matrix[rowIdx * size + colIdx] -= factorA * matrix[index * size + colIdx] / pivot;
    }

    __syncthreads();

    if (rowIdx < size && threadIdx.x == 0)
    {
        vector[rowIdx] -= matrix[rowIdx * size + index] * vector[index] / matrix[index * size + index];
        matrix[rowIdx * size + index] = 0.0;
    }
}

void init_data(double *matrix, double *vector, int size)
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

void back_substitute(double *matrix, double *vector, double *result, int size)
{
    result[size - 1] = vector[size - 1] / matrix[size * size - 1];
    for (int i = size - 2; i >= 0; --i)
    {
        double temp = vector[i];
        for (int j = i + 1; j < size; ++j)
        {
            temp -= matrix[i * size + j] * result[j];
        }
        result[i] = temp / matrix[i * size + i];
    }
}

int main()
{
    int size;
    cout << "Enter matrix size: ";
    cin >> size;

    double *matrix = new double[size * size];
    double *vector = new double[size];
    double *result = new double[size];

    init_data(matrix, vector, size);

    double *d_matrix, *d_vector;
    cudaMalloc(&d_matrix, sizeof(double) * size * size);
    cudaMalloc(&d_vector, sizeof(double) * size);

    cudaMemcpy(d_matrix, matrix, sizeof(double) * size * size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector, vector, sizeof(double) * size, cudaMemcpyHostToDevice);

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    float elapsed = 0.0f;

    cudaEventRecord(startEvent, 0);

    for (int i = 0; i < size - 1; ++i)
    {
        dim3 blockSize(256);
        dim3 gridSize(size - i - 1);
        gaussian_kernel<<<gridSize, blockSize>>>(d_matrix, d_vector, i, size);
        cudaDeviceSynchronize();
    }

    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&elapsed, startEvent, stopEvent);
    cout << "[Strategy B] GPU Gaussian Elimination completed in: " << elapsed << " ms" << endl;

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    cudaMemcpy(matrix, d_matrix, sizeof(double) * size * size, cudaMemcpyDeviceToHost);
    cudaMemcpy(vector, d_vector, sizeof(double) * size, cudaMemcpyDeviceToHost);
    back_substitute(matrix, vector, result, size);

    cudaFree(d_matrix);
    cudaFree(d_vector);
    delete[] matrix;
    delete[] vector;
    delete[] result;

    return 0;
}
