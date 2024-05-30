// RGB2YUV.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include<opencv2/opencv.hpp>
#include<iostream>
#include <chrono> 
using namespace std::chrono;
using namespace std;
using namespace cv;
int main()
{
	try
	{
		Mat img = imread("rgray8.bmp");

		if (!img.data)
		{
			std::cout << "Image not loaded";
			return -1;
		}
		

		Mat RGB;
		
		Mat image_blurred_with_3x3_kernel;
		auto start = high_resolution_clock::now();
		GaussianBlur(img, image_blurred_with_3x3_kernel, Size(3, 3), 0);
		auto stop = high_resolution_clock::now();
		

		imshow("Display window", image_blurred_with_3x3_kernel);

		auto duration = duration_cast<milliseconds>(stop - start);

		cout << "Time taken by function: " << duration.count() << " milliseconds" << endl;

		waitKey();

		return 0;
	}
	catch (cv::Exception& e)
	{
		cerr << e.msg << endl; // output exception message
	}

}