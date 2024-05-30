#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>

using namespace cv;
using namespace std;

Mat linearizeImage(const Mat& img) {
    Mat linearized = img.clone();
    linearized.convertTo(linearized, CV_32F, 1.0 / 255.0);
    for (int y = 0; y < linearized.rows; y++) {
        for (int x = 0; x < linearized.cols; x++) {
            for (int c = 0; c < linearized.channels(); c++) {
                float val = linearized.at<Vec3f>(y, x)[c];
                if (val <= 0.04045)
                    val = val / 12.92;
                else
                    val = pow((val + 0.055) / 1.055, 2.4);
                linearized.at<Vec3f>(y, x)[c] = val;
            }
        }
    }
    linearized.convertTo(linearized, CV_8U, 255.0);
    return linearized;
}

int main() {
    Mat img = imread("input_image.jpg");
    if (img.empty()) {
        cerr << "Could not open or find the image!" << endl;
        return -1;
    }

    auto start = chrono::high_resolution_clock::now();
    Mat linearized = linearizeImage(img);
    auto stop = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = stop - start;
    cout << "Linearization Time: " << elapsed.count() << " seconds" << endl;
    imwrite("linearized_image.jpg", linearized);

    return 0;
}
