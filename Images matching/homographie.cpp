#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <iostream>

#include "image.h"


using namespace std;
using namespace cv;

const float inlier_threshold = 2.5f; // Distance threshold to identify inliers
const float nn_match_ratio = 0.8f;   // Nearest neighbor matching ratio
const float regression_ratio = 0.9f;

int main()
{
	Image<uchar> I1 = Image<uchar>(imread("../IMG_0045.JPG", CV_LOAD_IMAGE_GRAYSCALE));
	Image<uchar> I2 = Image<uchar>(imread("../IMG_0046.JPG", CV_LOAD_IMAGE_GRAYSCALE));
	
	namedWindow("I1", 1);
	namedWindow("I2", 1);
	namedWindow("I2bis", 1);
	namedWindow("matches", 1);
	namedWindow("final", 1);
	
	imshow("I1", I1);
	imshow("I2", I2);

	// Step1: Finding first matching points with discrimination on nearest neighbors


	Image<uchar> desc1, desc2;
	vector<KeyPoint> kpts1, kpts2;
	Ptr<AKAZE> akaze = AKAZE::create();
	akaze->detectAndCompute(I1, noArray(), kpts1, desc1);
	akaze->detectAndCompute(I2, noArray(), kpts2, desc2);

	BFMatcher matcher(NORM_HAMMING);
	vector< vector<DMatch> > nn_matches;	// Matching points
	matcher.knnMatch(desc1, desc2, nn_matches, 2);

	vector<KeyPoint> matched1, matched2, inliers1, inliers2;
	vector<Point2f> srcPoints, dstPoints;
	vector<DMatch> matches1to2, good_matches1to2;

	Mat H;

	for (size_t i = 0; i < nn_matches.size(); i++) {
		DMatch first = nn_matches[i][0];
		float dist1 = nn_matches[i][0].distance;
		float dist2 = nn_matches[i][1].distance;

		if (dist1 < nn_match_ratio * dist2) {	// Discrimination on nearest neighbors
			matched1.push_back(kpts1[first.queryIdx]);
			matched2.push_back(kpts2[first.trainIdx]);
		}
	}

	// Step2: 

	for (unsigned i = 0; i < matched1.size(); i++) {
		srcPoints.push_back(matched1[i].pt);
		dstPoints.push_back(matched2[i].pt);
	}


	Mat I1bis(I1.cols, I1.rows, CV_8U);
	Mat I2bis(I2.cols, I2.rows, CV_8U);
	warpPerspective(I2, I2bis, H, Size(I2bis.rows, I2bis.cols));

	imshow("I1bis", I1bis);
	
	Mat K(Size(I1.cols * 2, I1.rows+5), CV_8U);
	I1.copyTo(K(cv::Rect(0, 0, I1.cols, I1.rows)));
	I2bis.copyTo(K(cv::Rect(I1.cols, 5, I2bis.cols, I2bis.rows)));

	imshow("final", K);

	waitKey(0);
	return 0;
}
