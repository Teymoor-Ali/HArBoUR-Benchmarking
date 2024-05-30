__global__ void imageSubtractKernel(const uchar* img1, const uchar* img2, uchar* result, int totalPixels)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < totalPixels)
    {
        result[index] = img1[index] - img2[index];
    }
}

int main(int argc, char* argv[])
{
    try
    {
        cv::Mat h_img1 = cv::imread("boat.bmp", cv::IMREAD_GRAYSCALE);
        cv::Mat h_img2 = cv::imread("boat.bmp", cv::IMREAD_GRAYSCALE);

        int totalPixels = h_img1.rows * h_img1.cols;

        // Allocate memory on the device
        uchar* d_img1, *d_img2, *d_result;
        cudaMalloc(&d_img1, totalPixels * sizeof(uchar));
        cudaMalloc(&d_img2, totalPixels * sizeof(uchar));
        cudaMalloc(&d_result, totalPixels * sizeof(uchar));

        // Upload Images to device
        cudaMemcpy(d_img1, h_img1.data, totalPixels * sizeof(uchar), cudaMemcpyHostToDevice);
        cudaMemcpy(d_img2, h_img2.data, totalPixels * sizeof(uchar), cudaMemcpyHostToDevice);

        dim3 block(256);
        dim3 grid((totalPixels + block.x - 1) / block.x);

        cudaEvent_t start, end;
        cudaEventCreate(&start);
        cudaEventCreate(&end);

        cudaEventRecord(start, 0);

        imageSubtractKernel<<<grid, block>>>(d_img1, d_img2, d_result, totalPixels);
        cudaDeviceSynchronize();

        cudaEventRecord(end, 0);
        cudaEventSynchronize(end);

        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, start, end);

        cudaEventDestroy(start);
        cudaEventDestroy(end);

        // Download result to host
        uchar* h_result = new uchar[totalPixels];
        cudaMemcpy(h_result, d_result, totalPixels * sizeof(uchar), cudaMemcpyDeviceToHost);

        // Display Execution Time
        cout << "Elapsed time in milliseconds: " << elapsedTime << " ms" << endl;

        delete[] h_result;
        cudaFree(d_img1);
        cudaFree(d_img2);
        cudaFree(d_result);

        return 0;
    }
    catch (const std::exception&)
    {
    }
}