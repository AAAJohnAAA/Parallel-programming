#define N 1024  // 问题规模，可调整
#define BLOCK_SIZE 64  // 可改为 64, 128, 256, 512, 1024

double A[N][N], b[N], x[N];
double *d_A, *d_b;

void gaussian_elimination_gpu() {
    size_t matrixSize = sizeof(double) * N * N;
    size_t vectorSize = sizeof(double) * N;

    cudaMalloc(&d_A, matrixSize);
    cudaMalloc(&d_b, vectorSize);

    cudaMemcpy(d_A, A, matrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, vectorSize, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int k = 0; k < N; ++k) {
        int threadsPerBlock = BLOCK_SIZE;
        int blocksPerGrid = (N - k - 1 + threadsPerBlock - 1) / threadsPerBlock;

        division_kernel<<<1, N - k - 1>>>(d_A, d_b, k, N);  // 归一化第k行
        elimination_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_b, k, N);  // 消元
        cudaDeviceSynchronize();
    }

    cudaMemcpy(A, d_A, matrixSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(b, d_b, vectorSize, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Total GPU Time: %f ms\n", milliseconds);

    // 回代（host端）
    for (int i = N - 1; i >= 0; i--) {
        double sum = b[i];
        for (int j = i + 1; j < N; j++) {
            sum -= A[i][j] * x[j];
        }
        x[i] = sum / A[i][i];
    }

    cudaFree(d_A);
    cudaFree(d_b);
}
