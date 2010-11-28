#undef __SSE2__

#include <iostream>
#include <ctime>
#include <cv.h>
#include <highgui.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <math_functions.h>

#include <map>

#include "ant.h"
#include "cutil/cutil.h"
#include "imageOp.h"

using namespace std;

#define ANTS 1024
#define WIDTH 512
#define HEIGHT 512

#define error(msg) {\
			cudaThreadSynchronize();\
			cudaError_t err = cudaGetLastError();\
			if( cudaSuccess != err) {\
				fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n", \
					msg, __FILE__, __LINE__, cudaGetErrorString( err ) );\
					exit(EXIT_FAILURE);\
			}\
		}

__device__ position deviceImage[512][512];
__device__ float broj;

texture<float, 2, cudaReadModeElementType> imageValues;
texture<float, 2, cudaReadModeElementType> heuristics;

__global__ void init(float *values, size_t pitch, float maxValue){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	float *q = (float *)((char *)values + j * pitch) + i;
	*q /= maxValue;
	deviceImage[i][j].pheromone = 0.001;
	deviceImage[i][j].antCount = 0;
}

//__global__ void setHeuristics(float *heuristics, int pitch){
//	float tl, tm, tr;
//	float ml, mr;
//	float bl, bm, br;
//
//	int i = blockIdx.x * blockDim.x + threadIdx.x;
//	int j = blockIdx.y * blockDim.y + threadIdx.y;
//
//	float intens[4];
//	float current = tex2D(imageValues, i, j);
//
//	tl = (i - 1 >= 0 && j - 1 >= 0) ? tex2D(imageValues, i - 1, j - 1) : current;
//	br = (i + 1 <= HEIGHT && j + 1 <= WIDTH) ? tex2D(imageValues, i - 1, j - 1) : current;
//	tr = (i - 1 >= 0 && j + 1 <= WIDTH) ? tex2D(imageValues, i - 1, j + 1) : current;
//	bl = (i + 1 <= HEIGHT && j - 1 >= 0) ? tex2D(imageValues, i + 1, j - 1) : current;
//	tm = (i - 1 >= 0) ? tex2D(imageValues, i - 1, j) : current;
//	bm = (i + 1 < HEIGHT) ? tex2D(imageValues, i + 1, j) : current;
//	ml = (j - 1 >= 0) ? tex2D(imageValues, i, j - 1) : current;
//	mr = (j + 1 < WIDTH) ? tex2D(imageValues, i, j + 1) : current;
//
//	intens[0] = fabs(tl - br);
//	intens[1] = fabs(tr - bl);
//	intens[2] = fabs(ml - mr);
//	intens[3] = fabs(tm - bm);
//
//	float max = intens[0];
//	for (int k = 1; k < 4; ++k) {
//		max = max > intens[i] ? max : intens[i];
//	}
//	float *q = (float *)((char *)heuristics + j * pitch) + i;
//	*q = current * max;
//
//	int index = 0;
//	if (i - 1 >= 0 && j - 1 >= 0) deviceImage[i][j].neigh[index++] = &deviceImage[i-1][j-1];
//	if (i + 1 < HEIGHT && j + 1 < WIDTH) deviceImage[i][j].neigh[index++] = &deviceImage[i+1][j+1];
//	if (i - 1 >= 0 && j + 1 < WIDTH) deviceImage[i][j].neigh[index++] = &deviceImage[i-1][j+1];
//	if (i + 1 < HEIGHT && j - 1 >= 0) deviceImage[i][j].neigh[index++] = &deviceImage[i+1][j-1];
//	if (i - 1 >= 0) deviceImage[i][j].neigh[index++] = &deviceImage[i-1][j];
//	if (i + 1 < HEIGHT) deviceImage[i][j].neigh[index++] = &deviceImage[i+1][j];
//	if (j - 1 >= 0) deviceImage[i][j].neigh[index++] = &deviceImage[i][j-1];
//	if (j + 1 < WIDTH) deviceImage[i][j].neigh[index++] = &deviceImage[i][j+1];
//
//	deviceImage[i][j].neighCount = index;
//	broj = deviceImage[0][0].neighCount;
//}

__global__ void test(){
	broj = tex2D(imageValues, 0, 0);
}

int main(int argc, char **argv){
	string inputFileImage = "resources/lena512s.png";
	IplImage *inputIplImage = cvLoadImage(inputFileImage.c_str(), CV_LOAD_IMAGE_GRAYSCALE);

	//ucitavanje slike
	float hostImageValues[HEIGHT][WIDTH];
	float maxValue = 0;
	for (int i = 0; i < HEIGHT; i++){
		for (int j = 0; j < 512; ++j){
			hostImageValues[i][j] = cvGet2D(inputIplImage, i, j).val[0];
			maxValue = maxValue > hostImageValues[i][j] ? maxValue : hostImageValues[i][j];
		}
	}
	//kraj ucitavanja slike


	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks(HEIGHT / threadsPerBlock.x, WIDTH / threadsPerBlock.y);

	//normaliziranje slike sivih razina, inicijalizacija feromonskih tragova
	float *deviceImageValues;
	size_t pitch;
	cudaMallocPitch((void **)&deviceImageValues, &pitch, sizeof(float) * WIDTH, HEIGHT);
	cudaMemcpy(deviceImageValues, hostImageValues, sizeof(float) * HEIGHT * WIDTH, cudaMemcpyHostToDevice);
	init<<<numBlocks, threadsPerBlock>>>(deviceImageValues, pitch, maxValue);

	cudaArray *imageValuesArray;
	cudaChannelFormatDesc cd = imageValues.channelDesc;
	cudaMallocArray(&imageValuesArray, &cd, WIDTH, HEIGHT);
	cudaMemcpyToArray(imageValuesArray, 0, 0, deviceImageValues, sizeof(float) * HEIGHT * WIDTH, cudaMemcpyDeviceToDevice);
	imageValues.addressMode[0] = cudaAddressModeWrap;
	imageValues.addressMode[1] = cudaAddressModeWrap;
	imageValues.filterMode     = cudaFilterModePoint;
	imageValues.normalized     = false;
	cudaBindTextureToArray(&imageValues, imageValuesArray, &cd);
	test<<<1, 1>>>();

	float t;
	cudaMemcpyFromSymbol(&t, "broj", sizeof(float), 0, cudaMemcpyDeviceToHost);
	cout << t << endl;


	cvReleaseImage(&inputIplImage);
	cudaFree(deviceImageValues);
	cudaFreeArray(imageValuesArray);
	return 0;
}
