#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <cuda_runtime.h>
#include <iostream>

// Linearization function kernel
__global__ void linearizeKernel(const uchar3* src, uchar3* dst, int width, int height, float a, float b) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        uchar3 pixel = src[idx];

        dst[idx].x = min(max(pixel.x * a + b, 0.0f), 255.0f);
        dst[idx].y = min(max(pixel.y * a + b, 0.0f), 255.0f);
        dst[idx].z = min(max(pixel.z * a + b, 0.0f), 255.0f);
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

    // Define linearization parameters
    float a = 1.2f;
    float b = 10.0f;

    // Timing the operation
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Configure block and grid sizes
    dim3 block(16, 16);
    dim3 grid((src.cols + block.x - 1) / block.x, (src.rows + block.y - 1) / block.y);

    cudaEventRecord(start);
    linearizeKernel<<<grid, block>>>(d_src.ptr<uchar3>(), d_dst.ptr<uchar3>(), src.cols, src.rows, a, b);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Linearization Time: " << milliseconds << " ms" << std::endl;

    // Download the result to host
    cv::Mat result;
    d_dst.download(result);

    // Save the result image
    cv::imwrite("output_linearized.jpg", result);

    // Clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
