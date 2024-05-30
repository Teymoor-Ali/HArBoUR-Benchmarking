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
		

	

		Mat BoxFilt;
		auto start = high_resolution_clock::now();
		boxFilter(img, BoxFilt, -1, cv::Size(3, 3));
		auto stop = high_resolution_clock::now();
		
		imshow("Display window", BoxFilt);
	


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