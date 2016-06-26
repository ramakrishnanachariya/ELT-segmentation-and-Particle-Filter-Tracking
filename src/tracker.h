#ifndef TRACKER_H
#define TRACKER_H
#include <opencv/cv.h>

extern cv::Rect* sArea;
extern bool	areaDefined;
float probability(const cv::Mat &img, cv::Rect rect, const cv::MatND &refhist);
//calculate the histogram of an HSV image
cv::MatND getHistogramHSV(const cv::Mat &img);
void mouseEventHandler(int event, int x, int y,int flags, void *params);

#endif
