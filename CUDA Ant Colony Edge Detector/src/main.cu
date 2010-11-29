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


texture<float, 2, cudaReadModeElementType> imageValuesTexture;
texture<float, 2, cudaReadModeElementType> heuristicsTexture;

__device__ int myRand(unsigned long seed){
	unsigned long next = seed * 1103515245 + 12345;
	unsigned long temp = ((unsigned)(next/65536) % 32768);
//	return (float)temp/32768;
	return temp;
}

__global__ void init(float *values, size_t pitch, float maxValue){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	float *q = (float *)((char *)values + j * pitch) + i;
	*q /= maxValue;
	deviceImage[i][j].pheromone = 0.001;
	deviceImage[i][j].antCount = 0;
}

__global__ void setHeuristics(float *heuristics, int pitch){
	float tl, tm, tr;
	float ml, mr;
	float bl, bm, br;

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	float intens[4];
	float current = tex2D(imageValuesTexture, i, j);

	tl = (i - 1 >= 0 && j - 1 >= 0) ? tex2D(imageValuesTexture, i - 1, j - 1) : current;
	br = (i + 1 <= HEIGHT && j + 1 <= WIDTH) ? tex2D(imageValuesTexture, i - 1, j - 1) : current;
	tr = (i - 1 >= 0 && j + 1 <= WIDTH) ? tex2D(imageValuesTexture, i - 1, j + 1) : current;
	bl = (i + 1 <= HEIGHT && j - 1 >= 0) ? tex2D(imageValuesTexture, i + 1, j - 1) : current;
	tm = (i - 1 >= 0) ? tex2D(imageValuesTexture, i - 1, j) : current;
	bm = (i + 1 < HEIGHT) ? tex2D(imageValuesTexture, i + 1, j) : current;
	ml = (j - 1 >= 0) ? tex2D(imageValuesTexture, i, j - 1) : current;
	mr = (j + 1 < WIDTH) ? tex2D(imageValuesTexture, i, j + 1) : current;


	intens[0] = fabs(tl - br);
	intens[1] = fabs(tr - bl);
	intens[2] = fabs(ml - mr);
	intens[3] = fabs(tm - bm);

	float max = intens[0];
	for (int k = 1; k < 4; ++k) {
		max = max > intens[k] ? max : intens[k];
	}

	float *currentHeuristicValue = (float *)((char *)heuristics + j * pitch) + i;
	*currentHeuristicValue = current * max;

	int index = 0;
	if (i - 1 >= 0 && j - 1 >= 0) deviceImage[i][j].neigh[index++] = &deviceImage[i-1][j-1];
	if (i + 1 < HEIGHT && j + 1 < WIDTH) deviceImage[i][j].neigh[index++] = &deviceImage[i+1][j+1];
	if (i - 1 >= 0 && j + 1 < WIDTH) deviceImage[i][j].neigh[index++] = &deviceImage[i-1][j+1];
	if (i + 1 < HEIGHT && j - 1 >= 0) deviceImage[i][j].neigh[index++] = &deviceImage[i+1][j-1];
	if (i - 1 >= 0) deviceImage[i][j].neigh[index++] = &deviceImage[i-1][j];
	if (i + 1 < HEIGHT) deviceImage[i][j].neigh[index++] = &deviceImage[i+1][j];
	if (j - 1 >= 0) deviceImage[i][j].neigh[index++] = &deviceImage[i][j-1];
	if (j + 1 < WIDTH) deviceImage[i][j].neigh[index++] = &deviceImage[i][j+1];

	deviceImage[i][j].neighCount = index;
}

__device__ int indeksi[1024];
__global__ void setAnts(ant *ants, unsigned long seed){
	int antIndex = blockDim.x * blockIdx.x + threadIdx.x;
	int currentSeed = (seed + antIndex) << 5;
	int randIndex = myRand(currentSeed) % ANTS;

	int i = randIndex / 32;
	int j = randIndex % 32;
	//atomic compare and swap
	while (atomicCAS(&deviceImage[i][j].antCount, 1, 1)){
		currentSeed <<= 1;
		randIndex = myRand(currentSeed) % ANTS;
	}
	atomicAdd(&deviceImage[i][j].antCount, 1);
	indeksi[antIndex] = randIndex;
}

__global__ void test(){
	broj = tex2D(heuristicsTexture, 50,50);
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
	float *deviceImageProperties;
	size_t pitch;

	cudaMallocPitch((void **)&deviceImageProperties, &pitch, sizeof(float) * WIDTH, HEIGHT);
	cudaMemcpy(deviceImageProperties, hostImageValues, sizeof(float) * HEIGHT * WIDTH, cudaMemcpyHostToDevice);

	init<<<numBlocks, threadsPerBlock>>>(deviceImageProperties, pitch, maxValue);
	//kraj inicijalizacije

	//"bindanje" matrice sivih razina u memorijski dio za texture (konstanta memorija brza od globalne)
	cudaArray *imageValuesArray;
	cudaChannelFormatDesc cd = imageValuesTexture.channelDesc;

	cudaMallocArray(&imageValuesArray, &cd, WIDTH, HEIGHT);
	cudaMemcpyToArray(imageValuesArray, 0, 0, deviceImageProperties, sizeof(float) * HEIGHT * WIDTH, cudaMemcpyDeviceToDevice);

	imageValuesTexture.addressMode[0] = cudaAddressModeWrap;
	imageValuesTexture.addressMode[1] = cudaAddressModeWrap;
	imageValuesTexture.filterMode     = cudaFilterModePoint;
	imageValuesTexture.normalized     = false;

	cudaBindTextureToArray(&imageValuesTexture, imageValuesArray, &cd);
	//kraj bindanja

	setHeuristics<<<numBlocks, threadsPerBlock>>>(deviceImageProperties, pitch);
	cudaArray *heuristicsArray;
	cd = heuristicsTexture.channelDesc;

	cudaMallocArray(&heuristicsArray, &cd, WIDTH, HEIGHT);
	cudaMemcpyToArray(heuristicsArray, 0, 0, deviceImageProperties, sizeof(float) * HEIGHT * WIDTH, cudaMemcpyDeviceToDevice);

	heuristicsTexture.addressMode[0] = cudaAddressModeWrap;
	heuristicsTexture.addressMode[1] = cudaAddressModeWrap;
	heuristicsTexture.filterMode     = cudaFilterModePoint;
	heuristicsTexture.normalized     = false;

	cudaBindTextureToArray(&heuristicsTexture, heuristicsArray, &cd);
	//kraj bindanja

	ant *ants;
	cudaMalloc(&ants, sizeof(ant) * ANTS);

//	setAnts<<<32, 32>>>(ants, (unsigned)time(0));



//	test<<<1, 1>>>();
//	int t[1024];
//	cudaMemcpyFromSymbol(t, "indeksi", sizeof(float) * 1024, 0, cudaMemcpyDeviceToHost);
//	for (int i = 0; i < ANTS; ++i)
//		cout << t[i] << " ";
//	cout << endl;



	cvReleaseImage(&inputIplImage);
	cudaFree(deviceImageProperties);
	cudaFree(ants);
	cudaFreeArray(imageValuesArray);
	cudaFreeArray(heuristicsArray);
	return 0;
}
