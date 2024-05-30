#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    // Load two images
    Mat img1 = imread("Rainier.bmp");
    Mat img2 = imread("Rainier.bmp");

    if (img1.empty() || img2.empty())
    {
        cerr << "Failed to load input image(s)" << endl;
        return -1;
    }

    // Declare SIFT detector and descriptor
    Ptr<Feature2D> sift = SIFT::create();

    // Detect keypoints and compute descriptors for both images
    vector<KeyPoint> kp1, kp2;
    Mat desc1, desc2;

    double start_time = (double)getTickCount();
    sift->detectAndCompute(img1, noArray(), kp1, desc1);
    double descriptor_generation_time = ((double)getTickCount() - start_time) / getTickFrequency();

    start_time = (double)getTickCount();
    sift->detectAndCompute(img2, noArray(), kp2, desc2);
    descriptor_generation_time += ((double)getTickCount() - start_time) / getTickFrequency();

    // Match descriptors
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    vector<vector<DMatch>> matches;

    start_time = (double)getTickCount();
    matcher->knnMatch(desc1, desc2, matches, 2);
    double matching_time = ((double)getTickCount() - start_time) / getTickFrequency();

    // Filter matches using Lowe's ratio test
    vector<DMatch> good_matches;
    const float ratio_threshold = 0.7f;
    for (size_t i = 0; i < matches.size(); ++i)
    {
        if (matches[i][0].distance < ratio_threshold * matches[i][1].distance)
            good_matches.push_back(matches[i][0]);
    }

    // Draw matches
    Mat img_matches;
    drawMatches(img1, kp1, img2, kp2, good_matches, img_matches);

    // Show results
    imshow("Matches", img_matches);

    // Compute total time
    double total_time = descriptor_generation_time + matching_time;
    cout << "Descriptor generation time: " << descriptor_generation_time << " seconds" << endl;
    cout << "Matching time: " << matching_time << " seconds" << endl;
    cout << "Total time: " << total_time << " seconds" << endl;

    waitKey(0);

    return 0;
}