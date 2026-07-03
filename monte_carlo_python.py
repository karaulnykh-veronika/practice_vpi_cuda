import numpy as np
import time

def monte_carlo_python(N):
    # Генерируем N случайных точек в квадрате [-1, 1]
    x = np.random.uniform(-1, 1, N)
    y = np.random.uniform(-1, 1, N)
    
    # Считаем, сколько попало в круг (x^2 + y^2 <= 1)
    M = np.sum(x*x + y*y <= 1)
    
    # Возвращаем приближенное значение Pi
    return 4.0 * M / N

if __name__ == "__main__":
    N = 500_000_000  # 500 миллионов  
    
    start = time.time()
    pi = monte_carlo_python(N)
    end = time.time()
    
    print(f"Число Pi (Python): {pi:.6f}")
    print(f"Время выполнения: {end - start:.3f} секунд")