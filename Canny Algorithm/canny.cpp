#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <queue>
#include <math.h> 

using namespace cv;
using namespace std;


void gradient(const Mat&Ic, Mat& Ix, Mat& Iy, Mat& G2)
{
	Mat I;
	cvtColor(Ic, I, CV_BGR2GRAY);

	int m = I.rows, n = I.cols;
	Ix = Mat(m, n, CV_32F);
	Iy = Mat(m, n, CV_32F);
	G2 = Mat(m, n, CV_32F);

	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			if (i == 0 || i == m - 1 || j == 0 || j == n - 1) {
				Ix.at<float>(i, j) = 0;
				Iy.at<float>(i, j) = 0;
				G2.at<float>(i, j) = 0;
			}
			else {
				Ix.at<float>(i, j) = ((float)I.at<uchar>(i + 1, j) - (float)I.at<uchar>(i - 1, j)) / 2;
				Iy.at<float>(i, j) = ((float)I.at<uchar>(i, j + 1) - (float)I.at<uchar>(i, j - 1)) / 2;
				G2.at<float>(i, j) = Ix.at<float>(i, j)*Ix.at<float>(i, j) + Iy.at<float>(i, j)*Iy.at<float>(i, j);
			}
		}
	}
}

void sobel(const Mat&Ic, Mat& Ix, Mat& Iy, Mat& G2)
{
	Mat I;
	cvtColor(Ic, I, CV_BGR2GRAY);

	int m = I.rows, n = I.cols;
	Ix = Mat(m, n, CV_32F);
	Iy = Mat(m, n, CV_32F);
	G2 = Mat(m, n, CV_32F);

	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			if (i == 0 || i == m - 1 || j == 0 || j == n - 1) {
				Ix.at<float>(i, j) = 0;
				Iy.at<float>(i, j) = 0;
				G2.at<float>(i, j) = 0;
			}
			else {
				Ix.at<float>(i, j) = (((float)I.at<uchar>(i + 1, j + 1) - (float)I.at<uchar>(i - 1, j + 1)) + 2 * ((float)I.at<uchar>(i + 1, j) - (float)I.at<uchar>(i - 1, j)) + ((float)I.at<uchar>(i + 1, j - 1) - (float)I.at<uchar>(i - 1, j - 1))) / 4;
				Iy.at<float>(i, j) = (((float)I.at<uchar>(i + 1, j + 1) - (float)I.at<uchar>(i + 1, j - 1)) + 2 * ((float)I.at<uchar>(i, j + 1) - (float)I.at<uchar>(i, j - 1)) + ((float)I.at<uchar>(i - 1, j + 1) - (float)I.at<uchar>(i - 1, j - 1))) / 4;
				G2.at<float>(i, j) = Ix.at<float>(i, j)*Ix.at<float>(i, j) + Iy.at<float>(i, j)*Iy.at<float>(i, j);
			}
		}
	}
}

Mat threshold(const Mat& Ic, float s, bool denoise = false)
{
	Mat Ix, Iy, G2;
	if (denoise)
		sobel(Ic, Ix, Iy, G2);
	else
		gradient(Ic, Ix, Iy, G2);
	int m = Ic.rows, n = Ic.cols;
	Mat C(m, n, CV_8U);
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++)
			if (i == 0 || i == m - 1 || j == 0 || j == n - 1 || G2.at<float>(i, j)<s) C.at<uchar>(i, j) = 0;
			else {
				float x = Ix.at<float>(i, j);
				float y = Iy.at<float>(i, j);
				if (x < 0) { x = -x; y = -y; }
				float r;
				if (x == 0) r = 3;
				else r = y / x;
				if (abs(r) > 2.41 && G2.at<float>(i, j)> G2.at<float>(i, j+1) && G2.at<float>(i, j)> G2.at<float>(i, j-1)) C.at<uchar>(i, j) = 255;
				else if (r > 0.41 && G2.at<float>(i, j) > G2.at<float>(i + 1, j+1) && G2.at<float>(i, j) > G2.at<float>(i-1, j-1)) C.at<uchar>(i, j) = 255;
				else if (r > -0.41 && G2.at<float>(i, j) > G2.at<float>(i+1, j) && G2.at<float>(i, j) > G2.at<float>(i-1, j)) C.at<uchar>(i, j) = 255;
				else if (G2.at<float>(i, j) > G2.at<float>(i+1, j-1) && G2.at<float>(i, j) > G2.at<float>(i-1, j+1)) C.at<uchar>(i, j) = 255;
				else C.at<uchar>(i, j) = 0;
			}
	return C;
}

Mat canny(const Mat& Ic, float s1)
{
	Mat Ix, Iy, G2;
	sobel(Ic, Ix, Iy, G2);

	int m = Ic.rows, n = Ic.cols;
	Mat Max(m, n, CV_8U);	
	Max = threshold(Ic, s1);
	queue<Point> Q;			
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			if (Max.at<uchar>(i, j) == 0 && G2.at<float>(i, j) > 3 * s1)
				Q.push(Point(j, i)); 

		}
	}

	Mat C(m, n, CV_8U);
	C.setTo(0);
	while (!Q.empty()) {
		int i = Q.front().y, j = Q.front().x;
		Q.pop();
		C.at<uchar>(i, j) = 255;
		if (Max.at<uchar>(i, j+1) == 255 && C.at<uchar>(i, j+1) == 0) {
			C.at<uchar>(i, j+1) = 255;
			Q.push(Point(j+1, i));
		}
		if (Max.at<uchar>(i, j-1) == 255 && C.at<uchar>(i, j-1) == 0) {
			C.at<uchar>(i, j-1) = 255;
			Q.push(Point(j-1, i));
		}
		if (Max.at<uchar>(i + 1, j) == 255 && C.at<uchar>(i + 1, j) == 0) {
			C.at<uchar>(i + 1, j) = 255;
			Q.push(Point(j, i + 1));
		}
		if (Max.at<uchar>(i + 1, j - 1) == 255 && C.at<uchar>(i + 1, j - 1) == 0) {
			C.at<uchar>(i + 1, j - 1) = 255;
			Q.push(Point(j - 1, i + 1));
		}
		if (Max.at<uchar>(i + 1, j + 1) == 255 && C.at<uchar>(i + 1, j + 1) == 0) {
			C.at<uchar>(i + 1, j + 1) = 255;
			Q.push(Point(j + 1, i + 1));
		}
		if (Max.at<uchar>(i - 1, j) == 255 && C.at<uchar>(i - 1, j) == 0) {
			C.at<uchar>(i - 1, j) = 255;
			Q.push(Point(j, i - 1));
		}
		if (Max.at<uchar>(i - 1, j - 1) == 255 && C.at<uchar>(i - 1, j - 1) == 0) {
			C.at<uchar>(i - 1, j - 1) = 255;
			Q.push(Point(j - 1, i - 1));
		}
		if (Max.at<uchar>(i - 1, j + 1) == 255 && C.at<uchar>(i - 1, j + 1) == 0) {
			C.at<uchar>(i - 1, j + 1) = 255;
			Q.push(Point(j + 1, i - 1));
		}
	}

	return C;
}

int main()
{
	Mat I = imread("../.jpg"); // add the path to your picture here
	float s = 500;
	imshow("Input", I);
	imshow("Threshold", threshold(I, s));
	imshow("Threshold + denoising", threshold(I, s, true));
	imshow("Canny", canny(I, s));

	waitKey();

	return 0;
}
