#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

int main() {
    // Просто проверяем, что CUDA работает
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "CUDA devices not found" << std::endl;
        return -1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    std::cout << "CUDA device: " << prop.name << std::endl;
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;

    // Засекаем время (имитация работы)
    auto start = std::chrono::high_resolution_clock::now();
    
    // Простая операция на GPU
    float* d_data;
    cudaMalloc(&d_data, 1024 * sizeof(float));
    cudaFree(d_data);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Время (CUDA): " << elapsed.count() << " с" << std::endl;

    return 0;
}