#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
using namespace std;

__global__ void gaussian_elimination_kernel(double *matrix, double *vector, int rowIndex, int matrixSize)
{
    int row = rowIndex + 1 + blockIdx.x;
    int col = rowIndex + 1 + threadIdx.x;

    if (row < matrixSize && col < matrixSize)
    {
        double factor = matrix[row * matrixSize + rowIndex];
        double pivot = matrix[rowIndex * matrixSize + rowIndex];
        matrix[row * matrixSize + col] -= factor * matrix[rowIndex * matrixSize + col] / pivot;
    }

    __syncthreads();

    if (row < matrixSize && threadIdx.x == 0)
    {
        vector[row] -= matrix[row * matrixSize + rowIndex] * vector[rowIndex] / matrix[rowIndex * matrixSize + rowIndex];
        matrix[row * matrixSize + rowIndex] = 0.0;
    }
}

void initialize_matrix_data(double *matrix, double *vector, int size)
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

void backward_substitution(double *matrix, double *vector, double *solution, int size)
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
    int matrixSize;
    cout << "Enter matrix size: ";
    cin >> matrixSize;

    int blockSizes[] = {64, 128, 256, 512, 1024};
    int numTests = sizeof(blockSizes) / sizeof(int);

    for (int testIndex = 0; testIndex < numTests; ++testIndex)
    {
        int blockSize = blockSizes[testIndex];

        double *matrix = new double[matrixSize * matrixSize];
        double *vector = new double[matrixSize];
        double *solution = new double[matrixSize];
        initialize_matrix_data(matrix, vector, matrixSize);

        double *d_matrix, *d_vector;
        cudaMalloc(&d_matrix, sizeof(double) * matrixSize * matrixSize);
        cudaMalloc(&d_vector, sizeof(double) * matrixSize);
        cudaMemcpy(d_matrix, matrix, sizeof(double) * matrixSize * matrixSize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_vector, vector, sizeof(double) * matrixSize, cudaMemcpyHostToDevice);

        cudaEvent_t startEvent, stopEvent;
        cudaEventCreate(&startEvent);
        cudaEventCreate(&stopEvent);
        float elapsedTime = 0.0;
        cudaEventRecord(startEvent, 0);

        for (int rowIndex = 0; rowIndex < matrixSize - 1; ++rowIndex)
        {
            int remainingColumns = matrixSize - (rowIndex + 1);
            int threadsForRow = min(blockSize, remainingColumns);
            dim3 blockDim(threadsForRow);
            dim3 gridDim(matrixSize - rowIndex - 1);
            gaussian_elimination_kernel<<<gridDim, blockDim>>>(d_matrix, d_vector, rowIndex, matrixSize);
            cudaDeviceSynchronize();
        }

        cudaEventRecord(stopEvent, 0);
        cudaEventSynchronize(stopEvent);
        cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
        cout << "[Block size = " << blockSize << "] GPU Gaussian elimination time: " << elapsedTime << " ms" << endl;

        cudaMemcpy(matrix, d_matrix, sizeof(double) * matrixSize * matrixSize, cudaMemcpyDeviceToHost);
        cudaMemcpy(vector, d_vector, sizeof(double) * matrixSize, cudaMemcpyDeviceToHost);
        backward_substitution(matrix, vector, solution, matrixSize);

        cudaFree(d_matrix);
        cudaFree(d_vector);
        delete[] matrix;
        delete[] vector;
        delete[] solution;
        cudaEventDestroy(startEvent);
        cudaEventDestroy(stopEvent);
    }

    return 0;
}
