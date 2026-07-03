import numpy as np
import time

def gauss_python(N):
    A = np.random.rand(N, N).astype(np.float32)
    b = np.random.rand(N).astype(np.float32)
    
    start = time.time()
    x = np.linalg.solve(A, b)
    end = time.time()
    
    return x, end - start

if __name__ == "__main__":
    sizes = [256, 512, 1024, 2048]
    for N in sizes:
        x, elapsed = gauss_python(N)
        print(f"Размер: {N}x{N}, Время: {elapsed:.6f} с")