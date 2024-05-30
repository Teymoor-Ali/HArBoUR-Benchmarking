#include <opencv2/opencv.hpp>
#include <opencv2/cudafilters.hpp>
#include <cuda_runtime.h>
#include <iostream>

int main() {
    // Load the image
    cv::Mat src = cv::imread("input_image.jpg", cv::IMREAD_COLOR);
    if (src.empty()) {
        std::cerr << "Error: Unable to load image!" << std::endl;
        return -1;
    }

    // Convert to GPU Mat
    cv::cuda::GpuMat d_src(src);
    cv::cuda::GpuMat d_dst;

    // Create a median filter
    cv::Ptr<cv::cuda::Filter> medianFilter = cv::cuda::createMedianFilter(d_src.type(), 5);

    // Timing the operation
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    medianFilter->apply(d_src, d_dst);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Median Filter Time: " << milliseconds << " ms" << std::endl;

    // Download the result to host
    cv::Mat result;
    d_dst.download(result);

    // Save the result image
    cv::imwrite("output_median_filter.jpg", result);

    // Clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
