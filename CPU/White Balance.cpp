#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>

using namespace cv;
using namespace std;

Mat whiteBalance(const Mat& img) {
    Mat balanced = img.clone();
    vector<Mat> channels;
    split(img, channels);
    Scalar meanVal = mean(img);
    for (int c = 0; c < 3; c++) {
        channels[c] = channels[c] * (meanVal[1] / meanVal[c]);
    }
    merge(channels, balanced);
    return balanced;
}

int main() {
    Mat img = imread("input_image.jpg");
    if (img.empty()) {
        cerr << "Could not open or find the image!" << endl;
        return -1;
    }

    auto start = chrono::high_resolution_clock::now();
    Mat white_balanced = whiteBalance(img);
    auto stop = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = stop - start;
    cout << "White Balance Time: " << elapsed.count() << " seconds" << endl;
    imwrite("white_balanced_image.jpg", white_balanced);

    return 0;
}
