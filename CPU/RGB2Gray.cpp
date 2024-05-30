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
		auto start = high_resolution_clock::now();
		
		Mat img = imread("Rainier.bmp");

		Mat RGB;

		Mat image_blurred_with_3x3_kernel;
		
		cvtColor(img, image_blurred_with_3x3_kernel, COLOR_RGB2GRAY);
		


		//imshow("Display window", image_blurred_with_3x3_kernel);
		imwrite("COLOUR.bmp", image_blurred_with_3x3_kernel);

		auto stop = high_resolution_clock::now();


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