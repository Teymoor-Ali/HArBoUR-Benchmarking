#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>

using namespace cv;
using namespace std;

Mat gammaCorrection(const Mat& img, float gamma) {
    Mat lookUpTable(1, 256, CV_8U);
    uchar* p = lookUpTable.ptr();
    for (int i = 0; i < 256; ++i)
        p[i] = saturate_cast<uchar>(pow(i / 255.0, gamma) * 255.0);

    Mat res = img.clone();
    LUT(img, lookUpTable, res);
    return res;
}

int main() {
    Mat img = imread("input_image.jpg");
    if (img.empty()) {
        cerr << "Could not open or find the image!" << endl;
        return -1;
    }

    auto start = chrono::high_resolution_clock::now();
    Mat gamma_corrected = gammaCorrection(img, 2.2f);
    auto stop = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = stop - start;
    cout << "Gamma Correction Time: " << elapsed.count() << " seconds" << endl;
    imwrite("gamma_corrected_image.jpg", gamma_corrected);

    return 0;
}
