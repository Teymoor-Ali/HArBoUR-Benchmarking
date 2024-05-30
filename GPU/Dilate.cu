#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
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

    // Upload the image to the GPU
    cv::cuda::GpuMat d_src, d_dst;
    d_src.upload(src);

    // Create CUDA dilate filter
    cv::Ptr<cv::cuda::Filter> dilate_filter = cv::cuda::createMorphologyFilter(cv::MORPH_DILATE, d_src.type(), element);

    // Start time measurement
    auto start = std::chrono::high_resolution_clock::now();

    // Apply dilation
    dilate_filter->apply(d_src, d_dst);

    // End time measurement
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Download the result to the CPU
    cv::Mat dilated_image;
    d_dst.download(dilated_image);

    // Display the results
    std::cout << "CUDA Dilation Time: " << elapsed.count() << " seconds" << std::endl;

    // Save the result
    cv::imwrite("dilated_image_cuda.jpg", dilated_image);

    return 0;
}
