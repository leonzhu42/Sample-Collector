#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "cv.h"
#include "asmfitting.h"
#include "asmlibrary.h"
#include "vjfacedetect.h"

#include <iostream>
#include <string>
#include <vector>
#include <strstream>
#include <fstream>
#include <cstring>

#include "asmfitting.h"
#include "asmlibrary.h"
#include "vjfacedetect.h"

using namespace cv;
using namespace std;

IplImage *image;
int fatigue_value = 5;
int prev_fatigue_value = fatigue_value;
const int n_iteration = 20;
const char *model_name = "my68-1d.amf";
const char *cascade_name = "haarcascade_frontalface_alt2.xml";
asmfitting fit_asm;
Mat frame;
CascadeClassifier face_cascade;
asm_shape shape, detshape;
RNG rng(12345);

string filenameGen()
{
	string name = "data/";
	if(fatigue_value < 10)
		name += "0";
	ostringstream strs;
	strs << fatigue_value;
	string num = strs.str();
	name += num;
	name += ' ';
	time_t rawtime;
	time(&rawtime);
	name += ctime(&rawtime);
	return name;
}

void DrawResult(IplImage* image, const asm_shape& shape)
{
	for(int i = 0; i < shape.NPoints(); ++i)
		cvCircle(image, cvPoint(shape[i].x, shape[i].y), 2, CV_RGB(255, 0, 0));
	Mat imageMat(image);
	imshow("ASM", imageMat);
}

void ASM_Save()
{
	//ASM
	IplImage *image = new IplImage(frame);
	bool flag = detect_one_face(detshape, image);
	if(flag)
		InitShapeFromDetBox(shape, detshape, fit_asm.GetMappingDetShape(), fit_asm.GetMeanFaceWidth());
	else
		return;
	fit_asm.Fitting(shape, image, n_iteration);
	DrawResult(image, shape);

	//Modify
	int width = shape.GetWidth();
	int height = shape.GetHeight();
	for(int i = 0; i < shape.NPoints(); ++i)
	{
		shape[i].x -= shape[shape.NPoints() - 1].x;
		shape[i].y -= shape[shape.NPoints() - 1].y;
		shape[i].x = shape[i].x * 200 / width;
		shape[i].y = shape[i].y * 200 / height;
	}

	//Save
	ofstream result(filenameGen().c_str());
	for(int i = 0; i < shape.NPoints(); ++i)
		result << shape[i].x << " " << shape[i].y << "\n";
	result.close();
}

void callbackCapture(int, void*)
{
	CvCapture* capture = cvCaptureFromCAM(-1);
	if(!capture)
		return;
	for(int i = 0; i < 10; ++i)
	{
		frame = cvQueryFrame(capture);
		if(frame.empty())
			continue;
		ASM_Save();
		waitKey(1000);
	}
}

void onFatigueValueChanged(int, void*)
{
	int pos = getTrackbarPos("Fatigue Value", "ASM");
	if(pos == 0)
		setTrackbarPos("Fatigue Value", "ASM", prev_fatigue_value);
	else
		prev_fatigue_value = fatigue_value = pos;
}

int main(int argc, char** argv)
{
	if(fit_asm.Read(model_name) == false)
		return -1;
	if(init_detect_cascade(cascade_name) == false)
		return -1;
	namedWindow("ASM");
	createTrackbar("Fatigue Value", "ASM", &fatigue_value, 10, onFatigueValueChanged);
	createButton("Capture", callbackCapture, NULL, CV_PUSH_BUTTON);
	waitKey(0);
	return 0;
}
