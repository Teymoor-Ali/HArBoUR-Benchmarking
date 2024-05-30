#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>

using namespace cv;
using namespace std;

Mat applyMedianFilter(const Mat& img, int ksize) {
    Mat filtered;
    medianBlur(img, filtered, ksize);
    return filtered;
}

int main() {
    Mat img = imread("input_image.jpg");
    if (img.empty()) {
        cerr << "Could not open or find the image!" << endl;
        return -1;
    }

    int ksize = 5; // Kernel size for the median filter
    auto start = chrono::high_resolution_clock::now();
    Mat median_filtered = applyMedianFilter(img, ksize);
    auto stop = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = stop - start;
    cout << "Median Filter Time: " << elapsed.count() << " seconds" << endl;
    imwrite("median_filtered_image.jpg", median_filtered);

    return 0;
}
