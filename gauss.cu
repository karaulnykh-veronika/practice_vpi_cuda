#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <iomanip>

__global__ void gaussKernel(float* A, float* b, int N, int col) {
    int row = blockIdx.x * blockDim.x + threadIdx.x + col + 1;
    if (row >= N) return;
    
    float factor = A[row * N + col] / A[col * N + col];
    
    for (int j = col; j < N; j++) {
        A[row * N + j] -= factor * A[col * N + j];
    }
    b[row] -= factor * b[col];
}

int main() {
    // Список размеров для тестирования
    int sizes[] = {256, 512, 1024, 2048};
    
    for (int idx = 0; idx < 4; idx++) {
        int N = sizes[idx];
        
        // Выделение памяти на CPU
        float* A = new float[N*N];
        float* b = new float[N];
        
        // Заполнение случайными числами
        for (int i = 0; i < N*N; i++) A[i] = rand() / (float)RAND_MAX;
        for (int i = 0; i < N; i++) b[i] = rand() / (float)RAND_MAX;
        
        // Выделение памяти на GPU
        float* d_A; cudaMalloc(&d_A, N*N*sizeof(float));
        float* d_b; cudaMalloc(&d_b, N*sizeof(float));
        
        // Копирование данных на GPU
        cudaMemcpy(d_A, A, N*N*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, N*sizeof(float), cudaMemcpyHostToDevice);
        
        int threads = 256;
        int blocks = (N + threads - 1) / threads;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // Прямой ход (исключение)
        for (int col = 0; col < N; col++) {
            gaussKernel<<<blocks, threads>>>(d_A, d_b, N, col);
            cudaDeviceSynchronize();
        }
        
        // Обратный ход (на CPU)
        float* x = new float[N];
        cudaMemcpy(A, d_A, N*N*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(b, d_b, N*sizeof(float), cudaMemcpyDeviceToHost);
        
        for (int i = N-1; i >= 0; i--) {
            float sum = 0.0f;
            for (int j = i+1; j < N; j++) {
                sum += A[i*N + j] * x[j];
            }
            x[i] = (b[i] - sum) / A[i*N + i];
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        
        // Вывод с 6 знаками после запятой (как в Python)
        std::cout << "Размер: " << N << "x" << N 
                  << ", Время: " << std::fixed << std::setprecision(6) 
                  << elapsed.count() << " с" << std::endl;
        
        // Очистка
        cudaFree(d_A); cudaFree(d_b);
        delete[] A; delete[] b; delete[] x;
    }
    
    return 0;
}