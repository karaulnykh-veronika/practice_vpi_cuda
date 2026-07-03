#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <vpi/VPI.h>
#include <vpi/Image.h>
#include <vpi/Status.h>
#include <vpi/OpenCVInterop.hpp>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return -1;
    }

    // 1. Загрузка изображения
    cv::Mat img = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Ошибка загрузки изображения" << std::endl;
        return -1;
    }

    // 2. Создание контекста VPI
    VPIContext context;
    vpiContextCreate(VPI_BACKEND_CUDA, &context);
    
    VPIStream stream;
    vpiStreamCreate(0, &stream);

    // 3. Создание VPI-изображений
    VPIImage vpiImg, vpiGray, vpiMedian, vpiSobel;
    vpiImageCreate(img.cols, img.rows, VPI_IMAGE_FORMAT_RGB8, 0, &vpiImg);
    vpiImageCreate(img.cols, img.rows, VPI_IMAGE_FORMAT_Y8, 0, &vpiGray);
    vpiImageCreate(img.cols, img.rows, VPI_IMAGE_FORMAT_Y8, 0, &vpiMedian);
    vpiImageCreate(img.cols, img.rows, VPI_IMAGE_FORMAT_Y8, 0, &vpiSobel);

    // 4. Копирование данных из OpenCV в VPI
    vpiImageCopyOpenCVMat(img, vpiImg);

    // 5. Засекаем время
    auto start = std::chrono::high_resolution_clock::now();

    // 6. Преобразование RGB → Grayscale
    vpiSubmitColorConvert(stream, 0, vpiImg, vpiGray, VPI_COLOR_RGB_TO_Y);

    // 7. Медианный фильтр (3x3)
    vpiSubmitMedianFilter(stream, 0, vpiGray, vpiMedian, 3, 0);

    // 8. Детектор Собеля (если функция есть)
    // Если нет — пропускаем, но результат будет без Sobel
    #ifdef VPI_HAS_FILTER_SOBEL
        vpiSubmitFilterSobel(stream, 0, vpiGray, vpiSobel, 3, 0);
    #else
        // Создаём заглушку, чтобы не было ошибки компиляции
        std::cout << "Sobel filter not available, skipping" << std::endl;
    #endif

    // 9. Ожидание завершения
    vpiStreamSync(stream);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // 10. Извлечение результатов из VPI в OpenCV
    cv::Mat gray, median;
    
    VPIImageData grayData;
    vpiImageLockData(vpiGray, VPI_LOCK_READ, &grayData);
    gray = cv::Mat(grayData.height, grayData.width, CV_8UC1, grayData.planes[0].data, grayData.planes[0].rowStride);
    vpiImageUnlockData(vpiGray);

    VPIImageData medianData;
    vpiImageLockData(vpiMedian, VPI_LOCK_READ, &medianData);
    median = cv::Mat(medianData.height, medianData.width, CV_8UC1, medianData.planes[0].data, medianData.planes[0].rowStride);
    vpiImageUnlockData(vpiMedian);

    // 11. Сохранение результатов
    cv::imwrite("vpi_gray.jpg", gray);
    cv::imwrite("vpi_median.jpg", median);

    std::cout << "Время (VPI): " << elapsed.count() << " с" << std::endl;
    std::cout << "Сохранено: vpi_gray.jpg, vpi_median.jpg" << std::endl;

    // 12. Очистка
    vpiImageDestroy(vpiImg);
    vpiImageDestroy(vpiGray);
    vpiImageDestroy(vpiMedian);
    vpiImageDestroy(vpiSobel);
    vpiStreamDestroy(stream);
    vpiContextDestroy(context);

    return 0;
}