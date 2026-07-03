#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <chrono>

// Ядро CUDA: каждая нить обрабатывает одну точку
__global__ void monteCarloKernel(int N, unsigned int seed, int* d_M) {
    // Индекс текущей нити (какая точка)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Инициализируем генератор случайных чисел для этой нити
    curandState state;
    curand_init(seed, idx, 0, &state);

    // Генерируем точку в квадрате [-1, 1]
    float x = curand_uniform(&state) * 2.0f - 1.0f;
    float y = curand_uniform(&state) * 2.0f - 1.0f;

    // Если точка внутри круга — увеличиваем глобальный счетчик
    if (x*x + y*y <= 1.0f) {
        atomicAdd(d_M, 1);
    }
}

int main() {
    int N = 500000000;  // 500 миллионов
    int M = 0;          // Счетчик попаданий (на CPU)
    int* d_M;           // Счетчик на GPU

    // Выделяем память на GPU
    cudaMalloc(&d_M, sizeof(int));
    cudaMemset(d_M, 0, sizeof(int));

    // Настраиваем сетку потоков
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Засекаем время
    auto start = std::chrono::high_resolution_clock::now();

    // Запускаем ядро на GPU
    monteCarloKernel<<<blocks, threadsPerBlock>>>(N, time(nullptr), d_M);
    cudaDeviceSynchronize();

    // Копируем результат обратно на CPU
    cudaMemcpy(&M, d_M, sizeof(int), cudaMemcpyDeviceToHost);

    // Засекаем время окончания
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Вычисляем Pi
    float pi = 4.0f * M / N;

    // Выводим результат
    std::cout << "Число Pi (CUDA): " << pi << std::endl;
    std::cout << "Время выполнения: " << elapsed.count() << " секунд" << std::endl;
    std::cout << "Попаданий: " << M << " из " << N << std::endl;

    // Очищаем память на GPU
    cudaFree(d_M);

    return 0;
}