

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <iostream>
#include "tracker.h"

using namespace cv;
using namespace std;

cv::Rect* sArea;
bool areaDefined = false;

MatND getHistogramHSV(const Mat &hsv) {
	
	MatND hist;
	int hbins = 30, sbins = 16;
    int histSize[] = {hbins, sbins};	
	float hranges[] = { 0, 180 };
	float sranges[] = { 0, 256 };
	const float* ranges[] = { hranges, sranges };
    // we compute the histogram from the 0-th and 1-st channels
    int channels[] = {0, 1};
	 calcHist( &hsv, 1, channels, Mat(), // do not use mask
             hist, 2, histSize, ranges,
             true, // the histogram is uniform
             false );

	//normalize the histogram, sum of the bins equal to one
	normalize(hist,hist,1.0,0,NORM_L1);
	return hist;
}
//used to define a rectangle selected with the mouse
void mouseEventHandler(int event, int x, int y,int flags, void *params) {
	static int x1,y1;
	static bool pressed=false;
	
	if(pressed) {
		rectangle(*(Mat*)params,Point(x1,y1),Point(x,y),Scalar(255,0,0) );
	}	
	if(event == CV_EVENT_LBUTTONDOWN) {
		x1 = x;
		y1 = y;
		pressed = true;	
	}
	if(event == CV_EVENT_LBUTTONUP) {
		sArea = new Rect(x1,y1,x-x1,y-y1);
		rectangle(*(Mat*)params,*sArea,Scalar(0,255,0) );
		pressed = false;
		areaDefined = true;
	}
}
