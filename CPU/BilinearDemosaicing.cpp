#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>

using namespace cv;
using namespace std;

Mat bilinearDemosaicing(const Mat& bayer_img) {
    Mat demosaiced;
    cvtColor(bayer_img, demosaiced, COLOR_BayerBG2BGR);
    return demosaiced;
}

int main() {
    Mat bayer_img = imread("input_bayer_image.jpg", IMREAD_GRAYSCALE);
    if (bayer_img.empty()) {
        cerr << "Could not open or find the image!" << endl;
        return -1;
    }

    auto start = chrono::high_resolution_clock::now();
    Mat demosaiced = bilinearDemosaicing(bayer_img);
    auto stop = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = stop - start;
    cout << "Bilinear Demosaicing Time: " << elapsed.count() << " seconds" << endl;
    imwrite("demosaiced_image.jpg", demosaiced);

    return 0;
}
