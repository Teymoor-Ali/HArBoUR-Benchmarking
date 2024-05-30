#include <iostream>
#include <chrono> 
#include "opencv2/opencv.hpp"

using namespace std::chrono;
using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
	auto start = chrono::high_resolution_clock::now();
	cv::cuda::GpuMat d_img1, d_result3x3;
	cv::Mat h_img1 = cv::imread("Rainier.bmp", 0);


	d_img1.upload(h_img1);
	cv::Ptr<cv::cuda::Filter> filter3x3, filter5x5, filter7x7;
	filter3x3 = cv::cuda::createBoxFilter(CV_8UC1, CV_8UC1, cv::Size(3, 3));
	filter3x3->apply(d_img1, d_result3x3);


	cv::Mat h_result3x3;
	
	d_result3x3.download(h_result3x3);
	auto end = chrono::high_resolution_clock::now();


	//Display Execution Time
	cout << "Elapsed time in nanoseconds : "
		<< chrono::duration_cast<chrono::nanoseconds>(end - start).count() * 10e-6
		<< " ms" << endl;

	cv::imshow("Result ", h_result3x3);
    cv::waitKey();
    return 0;