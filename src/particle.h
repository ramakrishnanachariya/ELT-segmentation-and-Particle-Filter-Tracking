#ifndef PARTICLES_H
#define PARTICLES_H
#include <opencv/cv.h>
#define SD  15
#define SCALE_SD 0.1
using namespace cv;
typedef struct part {
	//weight
	float w;
	//position information
	float x;
	float y;
	//scale information
	float scale;
	int width;
	int height;
	//histogram
	MatND *hist;
} particle;

extern RNG* rng;

particle* init_particles(Rect *region, MatND* hist, int regn, int partn);
void updateParticles(particle* pArr,int partn, Mat &hsvIm,const MatND &refHist,float sd,float scale_sd);
particle getBest(particle* pArr,int partn);
particle* resampleParticles(particle* pArr,int partn);
void sortParticles(particle* pArr, int partn);
#endif
