#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>

#include "maxflow/graph.h"

#include "image.h"

using namespace std;
using namespace cv;

/////////////////////////////////////////////////// 
//
//		        SOURCE
//		       /       \
//		     1/         \6
//		     /      4    \
//		   node0 -----> node1
//		     |   <-----   |
//		     |      3     |
//		     \            /
//		     5\          /1
//		       \        /
//		          SINK
//
///////////////////////////////////////////////////

int main() {

	Image<Vec3b> I= Image<Vec3b>(imread("../text.jpg"));
	//imshow("I",I);
	//waitKey(0);
	Image<uchar>G;
	Image<float> F;
	cvtColor(I, G, CV_BGR2GRAY);
	G.convertTo(F, CV_32F);


	int n = I.rows;
	int m = I.cols;


	typedef Graph<float, float, float> GraphType;
	GraphType *g = new GraphType(/*estimated # of nodes*/ n*m , /*estimated # of edges*/ 6 * (n*m + 2));

	g->add_node(n*m);

	float min = 1.;
	float max=0.;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			if (F.at<float>(i, j) < min) min = F.at<float>(i, j);
			if (F.at<float>(i, j) > max) max = F.at<float>(i, j);
		}
	}

	
	// edges of the SOURCE and the SINK
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			g->add_tweights(m*i+j, abs(70 - F.at<float>(i, j)), abs(200-F.at<float>(i, j)));
		}
	}

	// other edges
	for (int i = 1; i < n-2; i++) {
		for (int j = 1; j < m - 2; j++) {
			int value = m * i + j;
			int value_right = value + 1;
			int value_down = value + m;
			float gradient = ( (F.at<float>(i + 1, j) - F.at<float>(i - 1, j) )*( F.at<float>(i + 1, j) - F.at<float>(i -1, j) ) + ( F.at<float>(i, j+1) - F.at<float>(i, j-1) )*( F.at<float>(i, j + 1) - F.at<float>(i, j - 1))) / 4;
			float gradient_right = ((F.at<float>(i + 2, j) - F.at<float>(i, j))*(F.at<float>(i + 2, j) - F.at<float>(i, j)) + (F.at<float>(i+1, j + 1) - F.at<float>(i+1, j - 1))*(F.at<float>(i+1, j + 1) - F.at<float>(i+1, j - 1))) / 4;
			float gradient_down = ((F.at<float>(i + 1, j+1) - F.at<float>(i - 1, j+1))*(F.at<float>(i + 1, j+1) - F.at<float>(i - 1, j+1)) + (F.at<float>(i, j + 2) - F.at<float>(i, j))*(F.at<float>(i, j + 2) - F.at<float>(i, j))) / 4;
			g->add_edge(value, value_right, 1 / (1+gradient*gradient), 1 / (1 + gradient_right * gradient_right));
			g->add_edge(value, value_down, 1 / (1 + gradient * gradient), 1 / (1 + gradient_down * gradient_down));
			
		}
	}

	int flow = g->maxflow();

	cout << "Flow = " <<  flow << endl;


	for (int i = 0; i < n ; i++) {
		for (int j = 0; j < m; j++) {
			if (g->what_segment(m*i + j) == Graph<float, float, float>::SINK) {
				I.at<Vec3b>(i, j).val[0] = 0;
				I.at<Vec3b>(i, j).val[1] = 0;
				I.at<Vec3b>(i, j).val[2] = 0;
			}
			else {
				I.at<Vec3b>(i, j).val[0] = 255;
				I.at<Vec3b>(i, j).val[1] = 255;
				I.at<Vec3b>(i, j).val[2] = 255;
			}
		}
	}


	imshow("I", I);

	waitKey(0);

	delete g;

	return 0;

}
