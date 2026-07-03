import numpy as np
import time

def laplace_python(W, H, max_iter=1000, eps=1e-4):
    # Инициализация сетки
    T_old = np.zeros((H, W), dtype=np.float32)
    T_new = np.zeros((H, W), dtype=np.float32)
    
    # Источник тепла в центре
    center_y, center_x = H // 2, W // 2
    T_old[center_y-5:center_y+5, center_x-5:center_x+5] = 100.0
    T_new[center_y-5:center_y+5, center_x-5:center_x+5] = 100.0
    
    start = time.time()
    
    for it in range(max_iter):
        T_new[1:-1, 1:-1] = (T_old[:-2, 1:-1] + T_old[2:, 1:-1] + 
                              T_old[1:-1, :-2] + T_old[1:-1, 2:]) / 4.0
        
        diff = np.max(np.abs(T_new - T_old))
        T_old, T_new = T_new, T_old
        
        if diff < eps:
            break
    
    end = time.time()
    return it, end - start

if __name__ == "__main__":
    W, H = 1024, 1024
    iters, elapsed = laplace_python(W, H)
    print(f"Сетка: {W}x{H}")
    print(f"Итераций: {iters}")
    print(f"Время: {elapsed:.3f} с")