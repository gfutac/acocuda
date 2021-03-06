#undef __SSE2__

#include <iostream>
#include <time.h>
#include <cv.h>
#include <highgui.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "ant.h"
#include "imageOp.h"
#include "kernels.cu"

using namespace std;

int main(int argc, char **argv){
//	string inputFileImage = "resources/lena512s.png";
	string inputFileImage = argv[1];
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

	//definiranje velicina blokova dretvi
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

	//postavljanje generatora random brojeva
	curandState *devStates;
	cudaMalloc((void **)&devStates, sizeof(curandState) * ANTS);
	setupRandomsGenerator<<<32, 50>>>(devStates, (unsigned)time(NULL));
	//kraj postavljanja

	//postavljanje mrava na pocetne pozicije
	ant *ants;
	// 32 * 50 = 1600
	cudaMalloc(&ants, sizeof(ant) * ANTS);
	setAnts<<<32, 50>>>(ants, devStates);

	for (int i = 0; i < atoi(argv[2]); ++i){
		walk<<<32, 50>>>(ants, devStates);
		updateTrails<<<numBlocks, threadsPerBlock>>>();
	}

	position *slika = (position *)malloc(512 * 512 * sizeof(position));
	cudaMemcpyFromSymbol(slika, "deviceImage", 512 * 512 * sizeof(position), 0, cudaMemcpyDeviceToHost);
	IplImage *out = cvCreateImage(cvSize(WIDTH, HEIGHT), IPL_DEPTH_8U, 1);

	double total = 0;
	for (int i = 0; i < HEIGHT; ++i){
		for (int j = 0; j < WIDTH; ++j){
			total = total  + slika[i * WIDTH + j].pheromone;
		}
	}
	total /= (WIDTH * HEIGHT);

	for (int i = 0; i < HEIGHT; ++i){
		for (int j = 0; j < WIDTH; ++j){
			if (slika[i * WIDTH + j].pheromone > total) cvSet2D(out, j, i, cvScalar(255,0,0,0));
		}
	}

	showImage(out);

	free(slika);
	cvReleaseImage(&inputIplImage);
	cvReleaseImage(&out);
	cudaFree(deviceImageProperties);
	cudaFree(ants);
	cudaFree(devStates);
	cudaFreeArray(imageValuesArray);
	cudaFreeArray(heuristicsArray);
	return 0;
}
