import cv2
import time
import sys

if len(sys.argv) < 2:
    print("Usage: python3 image_python.py <image_path>")
    sys.exit(1)

img = cv2.imread(sys.argv[1])
if img is None:
    print("Ошибка загрузки")
    sys.exit(1)

start = time.time()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
median = cv2.medianBlur(gray, 3)
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
sobel = (sobel_x**2 + sobel_y**2)**0.5
sobel = (sobel * 255 / sobel.max()).astype('uint8')
end = time.time()

cv2.imwrite("python_gray.jpg", gray)
cv2.imwrite("python_median.jpg", median)
cv2.imwrite("python_sobel.jpg", sobel)

print(f"Время (Python): {end - start:.6f} с")