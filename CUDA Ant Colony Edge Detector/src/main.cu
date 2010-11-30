                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                +i){
			position *tmp = last->neigh[i];
			double probability = powf(tmp->pheromone, 4) * powf(tex2D(heuristicsTexture, tmp->i, tmp->j), 2);
			probabilities[i] = probability;
			probSum += probability;
		}
	}


	else if (ants[id].path.count > 1){
		position *penultimate = ants[id].path.penultimate();

		for (int i = 0; i < last->neighCount; ++i){
			if (ants[id].path.contains(last->neigh[i])) continue;
			if (last->neigh[i] == penultimate) continue;

			admissible[admissibleCount] = last->neigh[i];
			++admissibleCount;
		}

		for (int i = 0; i < admissibleCount; ++i){
			position *tmp = last->neigh[i];
			double probability = powf(tmp->pheromone, 4) * powf(tex2D(heuristicsTexture, tmp->i, tmp->j), 2);
			probabilities[i] = probability;
			probSum += probability;
		}
	}

	float r = curand_uniform(&localState) * probSum;
	float acumulatedSum = 0;
	position *next = 0;
	for (int i = 0; i < admissibleCount; ++i){
		acumulatedSum += probabilities[i];
		if (r < acumulatedSum) next = admissible[i];
	}
	if (!next){
		if (admissibleCount) next = admissible[--admissibleCount];
		else {
//			next = last;
		}
	}

	atomicAdd(&next->antCount, 1);
	ants[id].push_back(next);

	states[id] = localState;
}

__global__ void updateTrails(){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	float sum = 0;
	float heuristicValue = tex2D(heuristicsTexture, i, j);

	if (heuristicValue >= 0.08) {
		sum = heuristicValue * deviceImage[i][j].antCount;
	}
	deviceImage[i][j].pheromone = deviceImage[i][j].pheromone * (1 - 0.04) + sum;
	deviceImage[i][j].antCount = 0;
}

/******************************************************************************
*	RANDOM
******************************************************************************/
__global__ void setupRandomsGenerator(curandState *state, unsigned long seed){
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init((seed << 20) + id, id, 0, &state[id]);
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

	//postavljanje generatora random brojeva
	curandState *devStates;
	cudaMalloc((void **)&devStates, sizeof(curandState) * ANTS);
	setupRandomsGenerator<<<32, 32>>>(devStates, (unsigned)time(0));
	//kraj postavljanja

	//postavljanje mrava na pocetne pozicije
	ant *ants;
	cudaMalloc(&ants, sizeof(ant) * ANTS);
	setAnts<<<32, 32>>>(ants, devStates);

	for (int i = 0; i < 2; ++i){
		walk<<<32, 32>>>(ants, devStates);
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

	cout << total << endl;
	total /= (WIDTH * HEIGHT);

	for (int i = 0; i < HEIGHT; ++i){
		for (int j = 0; j < WIDTH; ++j){
			if (slika[i * WIDTH + j].pheromone >= total) cvSet2D(out, j, i, cvScalar(255,0,0,0));
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
