#include <iostream>
#include <vector>

__global__ void vectorAddKernel(float *vecA, float *vecB, float *vecC, int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
    {
        vecC[index] = vecA[index] + vecB[index];
    }
}

int main()
{
    int vectorSize = 1 << 20;
    size_t memorySize = vectorSize * sizeof(float);

    std::vector<float> hostVecA(vectorSize);
    std::vector<float> hostVecB(vectorSize);
    std::vector<float> hostVecC(vectorSize);

    for (int i = 0; i < vectorSize; ++i)
    {
        hostVecA[i] = static_cast<float>(i);
        hostVecB[i] = static_cast<float>(i * 2);
    }

    float *deviceVecA, *deviceVecB, *deviceVecC;

    cudaMalloc(&deviceVecA, memorySize);
    cudaMalloc(&deviceVecB, memorySize);
    cudaMalloc(&deviceVecC, memorySize);

    cudaMemcpy(deviceVecA, hostVecA.data(), memorySize, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceVecB, hostVecB.data(), memorySize, cudaMemcpyHostToDevice);

    int blockSize = 256;  
    int gridSize = (vectorSize + blockSize - 1) / blockSize;  
    vectorAddKernel<<<gridSize, blockSize>>>(deviceVecA, deviceVecB, deviceVecC, vectorSize);
    cudaMemcpy(hostVecC.data(), deviceVecC, memorySize, cudaMemcpyDeviceToHost);
    bool resultCorrect = true;
    for (int i = 0; i < vectorSize; ++i)
    {
        if (hostVecC[i] != hostVecA[i] + hostVecB[i])
        {
            resultCorrect = false;
            break;
        }
    }

    if (resultCorrect)
    {
        std::cout << "向量加法成功！" << std::endl;
    }
    else
    {
        std::cout << "向量加法失败！" << std::endl;
    }
    cudaFree(deviceVecA);
    cudaFree(deviceVecB);
    cudaFree(deviceVecC);

    return 0;
}
