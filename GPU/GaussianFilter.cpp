#include <iostream>
#include <opencv2/core/core.hpp> 
#include "opencv2/opencv.hpp"
#include "opencv2/core/cuda.hpp"
#include <opencv2/cudafilters.hpp>
#include <opencv2\imgproc\types_c.h>
#include <chrono> 
#include <cmath>
#include <opencv2\cudaimgproc.hpp>

using namespace std::chrono;
using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
	try
	{
		cv::Mat h_img1 = cv::imread("Rainier.bmp", 0);

		//Define device variables
		cv::cuda::GpuMat d_img1, d_result3x3;



		//Upload Image to device

		d_img1.upload(h_img1);

		auto start = chrono::steady_clock::now();

		cv::Ptr<cv::cuda::Filter> filter3x3;


		filter3x3 = cv::cuda::createGaussianFilter(CV_8UC1, CV_8UC1, cv::Size(3, 3), 0);

		filter3x3->apply(d_img1, d_result3x3);

		cv::Mat h_result3x3;
		auto end = chrono::steady_clock::now();
		//Download results back to host
		d_result3x3.download(h_result3x3);


		//Display Execution Time
		cout << "Elapsed time in nanoseconds : "
			<< chrono::duration_cast<chrono::nanoseconds>(end - start).count() * 10e-6
			<< " ms" << endl;


		cv::imshow("Result ", h_result3x3);
		cv::imwrite("gBlurred3x3.png", h_result3x3);
		return 0;
	}
	catch (const std::exception&)
	{
	}

}