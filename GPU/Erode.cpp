//erode

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
		cv::Mat h_img1 = cv::imread("boat.bmp", 0);

		// Define device variables
		cv::cuda::GpuMat d_img1, d_dilated;

		// Upload Image to device
		d_img1.upload(h_img1);

		auto start = chrono::steady_clock::now();

		cv::Ptr<cv::cuda::Filter> dilateFilter;

		dilateFilter = cv::cuda::createMorphologyFilter(cv::MORPH_DILATE, CV_8UC1, cv::Mat());

		dilateFilter->apply(d_img1, d_dilated);

		cv::Mat h_dilated;
		auto end = chrono::steady_clock::now();

		// Download results back to host
		d_dilated.download(h_dilated);

		// Display Execution Time
		cout << "Elapsed time in nanoseconds : "
			<< chrono::duration_cast<chrono::nanoseconds>(end - start).count() * 10e-6
			<< " ms" << endl;

		cv::imshow("Dilated Result", h_dilated);
		cv::imwrite("dilated_image.png", h_dilated);

		cv::waitKey(0);
		return 0;
	}
	catch (const std::exception&)
	{
	}

}
