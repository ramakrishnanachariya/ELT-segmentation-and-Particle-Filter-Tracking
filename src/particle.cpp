

#include "particle.h"
#include "tracker.h"
#include <opencv/cv.h>
#include <opencv/highgui.h>

using namespace cv;
using namespace std;
RNG *rng;

particle* init_particles(Rect *region, MatND* hist, int regn, int partn) {
	//class used to generate random numbers	
	rng = new RNG();
	//particles per region
	int ppr = floor(partn/regn);
	particle* pArr = (particle*)malloc(sizeof(particle)*partn);
	int pInd=0;
	int regX;
	int regY;


	for(int r=0; r < regn; r++) {
		//center of each region
		regX = region[r].x + region[r].width/2;
		regY = region[r].y + region[r].height/2;
		//init the particles of region r	
		for(int p=0; p < ppr; p++) {
			pArr[pInd].x = regX;
			pArr[pInd].y = regY;
			pArr[pInd].w  = 0;
			pArr[pInd].scale  = 1.0;
			pArr[pInd].height = region[r].height;
			pArr[pInd].width  = region[r].width;
			pArr[pInd++].hist =	new MatND(hist[r].clone()) ;			
		}
	}
	//loop until get exactly partn particles
	while( pInd < partn) {
		pArr[pInd].x = region[0].x + region[0].width/2;
		pArr[pInd].y = region[0].y + region[0].height/2;
		pArr[pInd].w  = 0;
		pArr[pInd].scale  = 1.0;
		pArr[pInd].height = region[0].height;
		pArr[pInd].width  = region[0].width;
		pArr[pInd++].hist =	new MatND(hist[0].clone());			
	}
	return pArr;
}

void updateParticles(particle* pArr,int partn, Mat &hsvIm,const MatND &refHist,float sd,float scale_sd) {
	int nx;
	int ny;
	float ns;
	int imgWidth = hsvIm.cols;
	int imgHeight = hsvIm.rows;
	MatND partHist;
	Rect *roi;
	float totalW=0;
	for (int p = 0; p < partn; p++) {
		//the new position of the particle is the current pos plus some gaussian generated number
		nx = rng->gaussian(sd) + pArr[p].x;
		ny = rng->gaussian(sd) + pArr[p].y;
		nx = max(1,min(nx,imgWidth-1));
		ny = max(1,min(ny,imgHeight-1));
		//the new scale is the current scale plus some gaussian generated number
		ns = rng->gaussian(scale_sd) + pArr[p].scale; 
		ns = abs(ns);
		ns = min(max(double(ns),0.1),(double)3);;
		//check if we are inside the image
		ns = ( (nx + (ns*pArr[p].width)/2) < (imgWidth-1) && (nx - (ns*pArr[p].width)/2) > 0 )?ns:0;
		ns = ( (ny + (ns*pArr[p].height)/2) < (imgHeight-1) && (ny - (ns*pArr[p].height)/2) > 0 )?ns:0;
		//if we are outside of the image dont update this particle
		if(ns==0) { 
			totalW += pArr[p].w;
			continue;
		}
		//assign new position and scale
		pArr[p].x  = nx;
		pArr[p].y  = ny;
		pArr[p].scale = ns;
		//get histogram of the particle
		roi = new Rect(nx - pArr[p].width*ns/2,
					   ny - pArr[p].height*ns/2,
					   pArr[p].width*ns,
					   pArr[p].height*ns);
		partHist = getHistogramHSV(hsvIm(*roi));
		//give weight based on histogram similarity with target 
		pArr[p].w = 1.0-compareHist(partHist,refHist,CV_COMP_BHATTACHARYYA);
		totalW   += pArr[p].w;
	}
	//normalize weights
	for(int p=0; p < partn; p++) {
		pArr[p].w = pArr[p].w/totalW;
	}
}
//select the particle with greater weight
particle getBest(particle* pArr,int partn) {
	particle part;
	part = pArr[0];
	for(int k=1; k < partn; k++) {
		if(pArr[k].w > part.w) {
			part = pArr[k];
		}
	}
	return part;
}

particle* resampleParticles(particle* pArr,int partn) {
	int pNum;
	int m=0;
	
	particle* newArr = (particle*)malloc(sizeof(particle)*partn);
	sortParticles(pArr,partn);
	//generate a new array containing two copies of the best particles
	for(int p=0; p < floor((float)partn/2); p++) {
		newArr[m++] = pArr[p];
		newArr[m++] = pArr[p];
	}	
	if(partn%2) {
		newArr[m] = newArr[m-1];
	}
	return newArr;
}
//order particles in acending order by weight
void sortParticles(particle* pArr, int partn) {
	particle temp;
	bool swapped;
	do {
		swapped = false;
		for(int p=1; p < partn; p++) {
			temp = pArr[p];
			if(pArr[p].w > pArr[p-1].w) {
				pArr[p] = pArr[p-1];
				pArr[p-1] = temp;
				swapped = true;
			} 
		}
	} while(swapped);
}
