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
		for (int i = 0; i < 100; i++) {

			
			cv::cuda::GpuMat d_img1, d_result3x3;
			cv::Mat h_img1 = cv::imread("Rainier.bmp", 0);
			auto start = chrono::high_resolution_clock::now();

			d_img1.upload(h_img1);
			cv::Ptr<cv::cuda::Filter> filter3x3, filter5x5, filter7x7;
			filter3x3 = cv::cuda::createBoxFilter(CV_8UC1, CV_8UC1, cv::Size(3, 3));
			filter3x3->apply(d_img1, d_result3x3);


			cv::Mat h_result3x3;
			auto end = chrono::high_resolution_clock::now();
			d_result3x3.download(h_result3x3);



			//Display Execution Time
			cout << "Elapsed time in nanoseconds : "
				<< chrono::duration_cast<chrono::nanoseconds>(end - start).count() * 10e-6
				<< " ms" << endl;

			cv::imshow("Result ", h_result3x3);
		}
		return 0;
	}
	catch (const std::exception&)
	{
	}

}