#undef __SSE2__

#include <stdio.h>
#include <time.h>
#include <cv.h>
#include <highgui.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <math_functions.h>

#include <map>

#include "ant.h"
#include "cutil/cutil.h"


//#include "cutil_inline.h"

using namespace std;

#define ANTS 1024

void showImage(IplImage *img){
	cvNamedWindow("Slika", CV_WINDOW_AUTOSIZE);
	cvMoveWindow("Slika", 440, 65);
	cvShowImage("Slika", img);
	while(true) if (cvWaitKey(10) == 27) break;
	cvDestroyWindow("Slika");
}

__device__ position d_imageMatrix[512][512];
__device__ float alpha = 4;
__device__ float beta = 2;
__device__ float sum[1024];


__device__ float getMax(float vals[4]){
	float max = vals[0];
	for (int i = 1; i < 4; ++i){
		max = max > vals[i] ? max : vals[i];
	}
	return max;
}

__device__ float myRand(unsigned long seed){
	unsigned long next = seed * 1103515245 + 12345;
	unsigned long temp = ((unsigned)(next/65536) % 32768);
	return (float)temp/32768;
}

__global__ void setImageValues(float *img, size_t pitch, float maxValue){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	float *q = (float *)((char *)img + j * pitch) + i;
	*q = (*q) / maxValue;

	d_imageMatrix[i][j].val = *q;
	d_imageMatrix[i][j].antCount = 0;
	d_imageMatrix[i][j].tao = 0.001;
}


__global__ void setNeighs(int height, int width){
    float tl, tm, tr;
    float ml, mr;
    float bl, bm, br;

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	float intens[4];
	tl = (i - 1 >= 0 && j - 1 >= 0) ? d_imageMatrix[i-1][j-1].val : d_imageMatrix[i][j].val;
	br = (i + 1 <= height && j + 1 <= width) ? d_imageMatrix[i+1][j+1].val : d_imageMatrix[i][j].val;
	tr = (i - 1 >= 0 && j + 1 <= width) ? d_imageMatrix[i-1][j+1].val : d_imageMatrix[i][j].val;
	bl = (i + 1 <= height && j - 1 >= 0) ? d_imageMatrix[i+1][j-1].val : d_imageMatrix[i][j].val;
	tm = (i - 1 >= 0) ? d_imageMatrix[i-1][j].val : d_imageMatrix[i][j].val;
	bm = (i + 1 < height) ? d_imageMatrix[i+1][j].val : d_imageMatrix[i][j].val;
	ml = (j - 1 >= 0) ? d_imageMatrix[i][j-1].val : d_imageMatrix[i][j].val;
	mr = (j + 1 < width) ? d_imageMatrix[i][j+1].val : d_imageMatrix[i][j].val;

	intens[0] = fabs(tl - br);
	intens[1] = fabs(tr - bl);
	intens[2] = fabs(ml - mr);
	intens[3] = fabs(tm - bm);

	d_imageMatrix[i][j].ni = d_imageMatrix[i][j].val * getMax(intens);

	int index = 0;
	if (i - 1 >= 0 && j - 1 >= 0) d_imageMatrix[i][j].neigh[index++] = &d_imageMatrix[i-1][j-1];
	if (i + 1 < height && j + 1 < width) d_imageMatrix[i][j].neigh[index++] = &d_imageMatrix[i+1][j+1];
	if (i - 1 >= 0 && j + 1 < width) d_imageMatrix[i][j].neigh[index++] = &d_imageMatrix[i-1][j+1];
	if (i + 1 < height && j - 1 >= 0) d_imageMatrix[i][j].neigh[index++] = &d_imageMatrix[i+1][j-1];
	if (i - 1 >= 0) d_imageMatrix[i][j].neigh[index++] = &d_imageMatrix[i-1][j];
	if (i + 1 < height) d_imageMatrix[i][j].neigh[index++] = &d_imageMatrix[i+1][j];
	if (j - 1 >= 0) d_imageMatrix[i][j].neigh[index++] = &d_imageMatrix[i][j-1];
	if (j + 1 < width) d_imageMatrix[i][j].neigh[index++] = &d_imageMatrix[i][j+1];

	d_imageMatrix[i][j].neighCount = index;
}

__global__ void setAnts(ant ants[ANTS]){
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	int x = ants[i].startx;
	int y = ants[i].starty;
	ants[i].push_back(&d_imageMatrix[x][y]);
}

__global__ void walk(ant ants[ANTS], unsigned long seed){
	int antIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (antIndex >= ANTS) return;
	
	int admissibleCount = 0;
	position *admissible[8];
	position *last = ants[antIndex].path.last();

	double probabilities[8];
	double probSum = 0;

	if (ants[antIndex].path.getCount() == 1){
		for (int i = 0; i < last->neighCount; ++i){
			admissible[i] = last->neigh[i];
			++admissibleCount;
		}

		for (int i = 0; i < admissibleCount; ++i){
			position *tmp = last->neigh[i];
			double probability = powf(tmp->tao, alpha) * powf(tmp->ni, beta);
			probabilities[i] = probability;
			probSum += probability;
		}
	}

	else if (ants[antIndex].path.getCount() > 1){
		position *penultimate = ants[antIndex].path.penultimate();
		
		for (int neighbors = 0; neighbors < last->neighCount; ++neighbors){
			if (ants[antIndex].path.contains(last->neigh[neighbors]) || (last->neigh[neighbors] == penultimate)) continue;
			admissible[admissibleCount++] = last->neigh[neighbors];
		}
		--admissibleCount;

		for (int i = 0; i < admissibleCount; ++i){
			position *tmp = admissible[i];
			double probability = powf(tmp->tao, alpha) * powf(tmp->ni, beta);
			probabilities[i] = probability;
			probSum += probability;
		}
	}

		
	double r = myRand(antIndex + seed) * probSum;
	double acumulatedSum = 0;
	position *next = 0;
	for (int i = 0; i < admissibleCount; ++i){
		acumulatedSum += probabilities[i];
		if (r < acumulatedSum) next = last->neigh[i];
	}
	if (!next){
		if (admissibleCount) next = admissible[admissibleCount];
		else {
			int index = (int)myRand(seed << 3) * 32786 % ants[antIndex].path.size();
			next = ants[antIndex].path[index];
		}
		//next = ants[antIndex].path[0];
	}

	atomicAdd(&next->antCount, 1);
	ants[antIndex].push_back(next);
}

__global__ void updateTrails(){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	float sum = 0;

	if (d_imageMatrix[i][j].ni >= 0.08) {
		sum = d_imageMatrix[i][j].ni * d_imageMatrix[i][j].antCount;
	}
	d_imageMatrix[i][j].tao = d_imageMatrix[i][j].tao * (1 - 0.04) + sum;
	d_imageMatrix[i][j].antCount = 0;
}


int main(int argc, char *argv[]){
	IplImage *in = cvLoadImage(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	int height = in->height;
	int width = in->width;
//	showImage(in);

	float maxValue = 0;
/*
	float *hostImageValues = (float *)malloc(height * width * sizeof(float));
	for (int i = 0; i < height; ++i){
		for (int j = 0; j < width; ++j){
			*(hostImageValues + i * width + j) = ((uchar *)(in->imageData + i*in->widthStep))[j];
			maxValue = maxValue > *(hostImageValues + i * width + j) ? maxValue : *(hostImageValues + i * width + j);
		}
	}
	
*/	
	float hostImageValues[512][512];
	for (int i = 0; i < height; ++i){
		for (int j = 0; j < width; ++j){
			hostImageValues[i][j] = cvGet2D(in, i, j).val[0];
			maxValue = maxValue > hostImageValues[i][j] ? maxValue : hostImageValues[i][j];
		}
	}

	float *imageIntensityValues;
	size_t pitch;
	cudaMallocPitch((void **)&imageIntensityValues, &pitch, sizeof(float) * width, height);
	cudaMemcpy(imageIntensityValues, hostImageValues, sizeof(float) * width * height, cudaMemcpyHostToDevice);

	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks(512 / threadsPerBlock.x, 512 / threadsPerBlock.y);
	setImageValues<<<numBlocks, threadsPerBlock>>>(imageIntensityValues, pitch, maxValue);
	setNeighs<<<numBlocks, threadsPerBlock>>>(height, width);

	map<pair<int, int>, bool> mapa;
	ant mravi[ANTS];
	srand((unsigned)time(NULL));
	int k = 0;
	while(k < ANTS){
		int i = rand() % height;
		int j = rand() % width;
		pair<int, int> lokacija (i, j);
		if (mapa.find(lokacija) != mapa.end()) continue;
		mapa[lokacija] = true;
		++k;
	}

	map<pair<int, int>, bool>::iterator it;
	int index = 0;
	for (it = mapa.begin(); it != mapa.end(); ++it){
		pair<int, int> p = it->first;
		int x = p.first;
		int y = p.second;
		int len = rand() % 15 + 25 + 1;
		mravi[index] = ant(len);
		mravi[index++].setStartPosition(x, y);
	}

	ant *deviceAnts;
	cudaMalloc((void **)&deviceAnts, ANTS * sizeof(ant));
	cudaMemcpy(deviceAnts, mravi, ANTS * sizeof(ant), cudaMemcpyHostToDevice);
	setAnts<<<32, 32>>>(deviceAnts);


	int iter = atoi(argv[2]);
	for (int i = 0; i < iter; ++i){
		walk<<<32, 32>>>(deviceAnts, (unsigned)time(NULL));
		updateTrails<<<numBlocks, threadsPerBlock>>>();
	}

	position *slika = (position *)malloc(512 * 512 * sizeof(position));
	cudaMemcpyFromSymbol(slika, "d_imageMatrix", 512 * 512 * sizeof(position), 0, cudaMemcpyDeviceToHost);

	IplImage *out = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);

	float total = 0;
	for (int i = 0; i < height; ++i){
		for (int j = 0; j < width; ++j){
			total = total  + slika[i * width + j].tao;
		}
	}
	
	printf("%f\n", total);
	total /= (width * height);

	for (int i = 0; i < height; ++i){
		for (int j = 0; j < width; ++j){
			if (slika[i * width + j].tao >= total) cvSet2D(out, j, i, cvScalar(255,0,0,0));
		}
	}

	cudaFree(imageIntensityValues);
	cudaFree(deviceAnts);
	//free(hostImageValues);

	showImage(out);
	cvReleaseImage(&out);
	cvReleaseImage(&in);

	free(slika);
    return 0;
}

