#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#define BLOCK_SIZE 16

__global__ void laplaceKernel(float* d_T_new, const float* d_T_old, int W, int H, float* d_diff) {
    __shared__ float shared[BLOCK_SIZE+2][BLOCK_SIZE+2];
    
    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int tx = threadIdx.x + 1;
    int ty = threadIdx.y + 1;
    
    // Загрузка данных в разделяемую память (с соседями)
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int sx = x + dx;
            int sy = y + dy;
            if (sx >= 0 && sx < W && sy >= 0 && sy < H) {
                shared[ty + dy][tx + dx] = d_T_old[sy * W + sx];
            } else {
                shared[ty + dy][tx + dx] = 0.0f; // граничные условия
            }
        }
    }
    __syncthreads();
    
    // Обновление узла (только внутренние)
    if (x > 0 && x < W-1 && y > 0 && y < H-1) {
        float val = (shared[ty-1][tx] + shared[ty+1][tx] + 
                     shared[ty][tx-1] + shared[ty][tx+1]) * 0.25f;
        d_T_new[y * W + x] = val;
        
        // Вычисление разницы для проверки сходимости
        float diff = fabs(val - d_T_old[y * W + x]);
        atomicMax((unsigned int*)d_diff, *((unsigned int*)&diff));
    }
}

int main() {
    int W = 1024, H = 1024;
    int max_iter = 1000;
    float eps = 1e-4;
    
    // Выделение памяти на CPU
    float* T_old = new float[W*H]();
    float* T_new = new float[W*H]();
    
    // Источник тепла
    int cy = H/2, cx = W/2;
    for (int dy = -5; dy < 5; dy++) {
        for (int dx = -5; dx < 5; dx++) {
            int x = cx + dx, y = cy + dy;
            if (x >= 0 && x < W && y >= 0 && y < H) {
                T_old[y*W + x] = 100.0f;
            }
        }
    }
    
    // Выделение памяти на GPU
    float* d_T_old; cudaMalloc(&d_T_old, W*H*sizeof(float));
    float* d_T_new; cudaMalloc(&d_T_new, W*H*sizeof(float));
    float* d_diff; cudaMalloc(&d_diff, sizeof(float));
    
    // Копирование данных на GPU
    cudaMemcpy(d_T_old, T_old, W*H*sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((W + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                (H + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    int iter;
    for (iter = 0; iter < max_iter; iter++) {
        cudaMemset(d_diff, 0, sizeof(float));
        
        laplaceKernel<<<blocks, threads>>>(d_T_new, d_T_old, W, H, d_diff);
        cudaDeviceSynchronize();
        
        float diff;
        cudaMemcpy(&diff, d_diff, sizeof(float), cudaMemcpyDeviceToHost);
        
        // Обмен буферами
        std::swap(d_T_old, d_T_new);
        
        if (diff < eps) break;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    
    // Копирование результата на CPU
    cudaMemcpy(T_old, d_T_old, W*H*sizeof(float), cudaMemcpyDeviceToHost);
    
    std::cout << "Сетка: " << W << "x" << H << std::endl;
    std::cout << "Итераций: " << iter << std::endl;
    std::cout << "Время: " << elapsed.count() << " с" << std::endl;
    
    // Очистка
    cudaFree(d_T_old); cudaFree(d_T_new); cudaFree(d_diff);
    delete[] T_old; delete[] T_new;
    
    return 0;
}