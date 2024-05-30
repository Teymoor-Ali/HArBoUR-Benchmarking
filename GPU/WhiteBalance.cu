#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <cuda_runtime.h>
#include <iostream>

// White balance function kernel
__global__ void whiteBalanceKernel(const uchar3* src, uchar3* dst, int width, int height, float r_gain, float g_gain, float b_gain) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        uchar3 pixel = src[idx];

        dst[idx].x = min(max(pixel.x * b_gain, 0.0f), 255.0f);
        dst[idx].y = min(max(pixel.y * g_gain, 0.0f), 255.0f);
        dst[idx].z = min(max(pixel.z * r_gain, 0.0f), 255.0f);
    }
}

int main() {
    // Load the image
    cv::Mat src = cv::imread("input_image.jpg", cv::IMREAD_COLOR);
    if (src.empty()) {
        std::cerr << "Error: Unable to load image!" << std::endl;
        return -1;
    }

    // Convert to GPU Mat
    cv::cuda::GpuMat d_src(src);
    cv::cuda::GpuMat d_dst(src.size(), src.type());

    // Define white balance gains
    float r_gain = 1.2f;
    float g_gain = 1.0f;
    float b_gain = 1.1f;

    // Timing the operation
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Configure block and grid sizes
    dim3 block(16, 16);
    dim3 grid((src.cols + block.x - 1) / block.x, (src.rows + block.y - 1) / block.y);

    cudaEventRecord(start);
    whiteBalanceKernel<<<grid, block>>>(d_src.ptr<uchar3>(), d_dst.ptr<uchar3>(), src.cols, src.rows, r_gain, g_gain, b_gain);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "White Balance Time: " << milliseconds << " ms" << std::endl;

    // Download the result to host
    cv::Mat result;
    d_dst.download(result);

    // Save the result image
    cv::imwrite("output_white_balance.jpg", result);

    // Clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
