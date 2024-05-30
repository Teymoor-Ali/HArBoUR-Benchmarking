
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

		Mat img = imread("Rainier.bmp");

		if (!img.data)
		{
			std::cout << "Image not loaded";
			return -1;
		}
		

		Mat RGB;
		auto start = high_resolution_clock::now();
		Mat image_blurred_with_3x3_kernel;
		Sobel(img, image_blurred_with_3x3_kernel, -1, 1, 1);
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