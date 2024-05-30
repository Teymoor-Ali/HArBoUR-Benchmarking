#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>

using namespace cv;
using namespace std;

Mat resizeImage(const Mat& img) {
    Mat resized;
    resize(img, resized, Size(), 0.5, 0.5);
    return resized;
}

int main() {
    Mat img = imread("input_image.jpg");
    if (img.empty()) {
        cerr << "Could not open or find the image!" << endl;
        return -1;
    }

    auto start = chrono::high_resolution_clock::now();
    Mat resized = resizeImage(img);
    auto stop = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = stop - start;
    cout << "Resizing Time: " << elapsed.count() << " seconds" << endl;
    imwrite("resized_image.jpg", resized);

    return 0;
}
