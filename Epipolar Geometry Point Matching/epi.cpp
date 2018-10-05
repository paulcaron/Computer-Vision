#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include "image.h"

struct Camera {
	Matx33d A;
	Vec3d b;
	void read(string name) {
		ifstream f;
		f.open(name);
		if (!f.is_open()) {
			cout << "Cannot read camera file" << endl;
			return;
		}
		for (int i = 0; i < 3; i++)
			f >> A(i, 0) >> A(i, 1) >> A(i, 2) >> b[i];
		f.close();
	}
	void print() const {
		cout << "A= " << A << endl
			<< "b= " << b << endl;
	}
	Vec3d center() const {
		return Vec3d((-A.inv()*b).val);
	}
	Vec3d proj(const Vec3d& M) const {
		return Vec3d((A*M + b).val);
	}
};

Matx33d fundamental(const Camera& C1, const Camera& C2) {
	Vec3d e2 = C2.proj(C1.center());
	Matx33d E2(0, -e2[2], e2[1],
		e2[2], 0, -e2[0],
		-e2[1], e2[0], 0);
	return E2 * C2.A*C1.A.inv();
}

struct Data {
	Image<Vec3b> I1, I2;
	Image<float> F1, F2;
	Camera C1, C2;
	Matx33d F;
};

void onMouse1(int event, int x, int y, int foo, void* p)
{
	if (event != CV_EVENT_LBUTTONDOWN)
		return;
	Point m1(x, y);

	Data* D = (Data*)p;
	circle(D->I1, m1, 2, Scalar(0, 255, 0), 2);
	imshow("I1", D->I1);

	Vec3d m1p(m1.x, m1.y, 1);
	// Epipolar line equation 
	Vec3d l = D->F*m1p;

	Point e2( D->C2.proj(D->C1.center())[0] , D->C2.proj(D->C1.center())[1]) ;
	Point m2a(0, -l[2]/l[1]);
	Point m2b(D->I2.width(), -( D->I2.width()*l[0] + l[2]) / l[1]);

	line(D->I2,m2a,m2b,Scalar(0,255,0),1);

	double val_min = 0.;
	Point pt = 0;
	LineIterator it(D->I2, m2a, m2b, 8);
	cout << "Ah" << endl;
	for (int i = 0; i < it.count; i++, ++it)
	{
		if (NCC(D->F1, m1, D->F2, it.pos(), 5) > val_min) {
			cout << i << endl;
			pt = it.pos();
			val_min = NCC(D->F1, m1, D->F2, it.pos(), 1);
		}
	}
	circle(D->I2, pt, 2, Scalar(0, 255, 0), 2);
	imshow("I2", D->I2);
}

void onMouse2(int event, int x, int y, int foo, void* p)
{
	
	if (event != CV_EVENT_LBUTTONDOWN)
		return;
	Point m2(x, y);

	Data* D = (Data*)p;
	circle(D->I2, m2, 2, Scalar(0, 0, 255), 2);
	imshow("I2", D->I2);

	Vec3d m2p(m2.x, m2.y, 1);
	Vec3d l = D->F.t()*m2p;

	Point e1(D->C1.proj(D->C2.center())[0], D->C1.proj(D->C2.center())[1]);
	Point m1a(0, -l[2] / l[1]);
	Point m1b(D->I1.width(), -(D->I1.width()*l[0] + l[2]) / l[1]);
	line(D->I1, m1a, m1b, Scalar(0, 0, 255), 1);


	double val_min = 0.;
	Point pt = 0;
	LineIterator it(D->I1, m1a, m1b, 8);
	for (int i = 0; i < it.count; i++, ++it)
	{
		if (NCC(D->F2, m2, D->F1, it.pos(), 5) > val_min) {
			cout << i << endl;
			pt = it.pos();
			val_min = NCC(D->F2, m2, D->F1, it.pos(), 1);
		}
	}
	circle(D->I1, pt, 2, Scalar(0, 0, 255), 2);
	imshow("I1", D->I1);

}

int main(int argc, char** argv)
{
	Data D;
	D.I1 = imread("../face00.tif");
	D.I2 = imread("../face01.tif");
	imshow("I1", D.I1);
	imshow("I2", D.I2);

	D.C1.read("../face00.txt");
	D.C2.read("../face01.txt");
	D.C1.print();
	D.C2.print();

	D.F = fundamental(D.C1, D.C2);
	cout << "F= " << D.F << endl;

	Image<uchar>G1, G2;
	cvtColor(D.I1, G1, CV_BGR2GRAY);
	cvtColor(D.I2, G2, CV_BGR2GRAY);
	G1.convertTo(D.F1, CV_32F);
	G2.convertTo(D.F2, CV_32F);

	setMouseCallback("I1", onMouse1, &D);
	setMouseCallback("I2", onMouse2, &D);

	waitKey(0);
	return 0;
}
