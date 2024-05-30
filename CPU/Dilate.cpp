#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

int main() {
    // Read the image
    cv::Mat src = cv::imread("input_image.jpg", cv::IMREAD_GRAYSCALE);
    if (src.empty()) {
        std::cerr << "Could not open or find the image!" << std::endl;
        return -1;
    }

    // Create a structuring element
    int dilation_size = 3;
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT,
                                                cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1),
                                                cv::Point(dilation_size, dilation_size));

    // Start time measurement
    auto start = std::chrono::high_resolution_clock::now();

    // Apply dilation
    cv::Mat dilated_image;
    cv::dilate(src, dilated_image, element);

    // End time measurement
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Display the results
    std::cout << "CPU Dilation Time: " << elapsed.count() << " seconds" << std::endl;

    // Save the result
    cv::imwrite("dilated_image_cpu.jpg", dilated_image);

    return 0;
}
