#include <iostream>
#include <chrono> 
#include "opencv2/opencv.hpp"

using namespace std::chrono;
using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{

	cv::Mat h_img1 = cv::imread("Rainier.bmp", 0);
	cv::cuda::GpuMat d_img1, d_resultx, d_resulty, d_resultxy;


	d_img1.upload(h_img1);

	auto start = chrono::high_resolution_clock::now();

	cv::Ptr<cv::cuda::Filter> filterx, filtery, filterxy;

	filterx = cv::cuda::createSobelFilter(CV_8UC1, CV_8UC1, 1, 0);

	filterx->apply(d_img1, d_resultx);

	filtery = cv::cuda::createSobelFilter(CV_8UC1, CV_8UC1, 0, 1);

	filtery->apply(d_img1, d_resulty);

	cv::cuda::add(d_resultx, d_resulty, d_resultxy);

	cv::Mat h_resultxy;
	auto end = chrono::steady_clock::now();

	d_resultxy.download(h_resultxy);


	cv::imwrite("sobelxy.png", h_resultxy);

	//Display Execution Time
	cout << "Elapsed time in nanoseconds : "
		<< chrono::duration_cast<chrono::nanoseconds>(end - start).count() * 10e-6
		<< " ms" << endl;

	cv::imshow("Result ", h_resultxy);
    cv::waitKey();
    return 0;
}