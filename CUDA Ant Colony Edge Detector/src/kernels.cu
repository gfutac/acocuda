#define ANTS 1600
#define WIDTH 512
#define HEIGHT 512


/*
 * Osnovna struktura koja sadrzi podatke o svakoj poziciji slike
 */
__device__ position deviceImage[HEIGHT][WIDTH];

/*
 * Dio memorijskog prostora za teksture namijenjen cuvanju vrijednosti sivih razni slike
 */
texture<float, 2, cudaReadModeElementType> imageValuesTexture;

/*
 * Dio memorijskog prostora za teksture namijenjen cuvanju heuristickih vrijednosti
 * svake pozicije
 */
texture<float, 2, cudaReadModeElementType> heuristicsTexture;

/*
 * Kernel koji inicjalizira generator slucajnih brojeva
 */
__global__ void setupRandomsGenerator(curandState *state, unsigned long int seed){
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed + id, id, 0, &state[id]);
}

/*
 * Kernel koji inicijalizira vrijednosti feromonskih tragova na pocetnu vrijednost
 */
__global__ void init(float *values, size_t pitch, float maxValue){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	float *q = (float *)((char *)values + j * pitch) + i;
	*q /= maxValue;
	deviceImage[i][j].pheromone = 0.001;
	deviceImage[i][j].antCount = 0;
	deviceImage[i][j].i = i;
	deviceImage[i][j].j = j;
}

/*
 * Kernel koji racuna vidljivost svake pozicije i odreduje susjedne pozicije
 * trenutno promatrane pozicije
 */
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

/*
 * Kernel koji inicijalizira pozicije mrava (odreduje pocetne polozaje)
 */
__global__ void setAnts(ant *ants, curandState *states){
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= ANTS) return;
	curandState localState = states[id];
	ants[id] = ant(curand(&localState) % 15 + 25 + 1);

	int i = curand(&localState) % HEIGHT;
	int j = curand(&localState) % WIDTH;

	while (atomicCAS(&deviceImage[i][j].antCount, 1, 1)){
		i = curand(&localState) % HEIGHT;
		j = curand(&localState) % WIDTH;
	}

	atomicAdd(&deviceImage[i][j].antCount, 1);
	ants[id].path.push_back(&deviceImage[i][j]);
	states[id] = localState;
}

/*
 * Kernel koji obavlja hod jednog mrav (odreduje sljedecu poziciju)
 */
__global__ void walk(ant *ants, curandState *states){
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= ANTS) return;
	int admissibleCount = 0;

	position *admissible[8];
	position *last = ants[id].path.last();

	double probabilities[8];
	double probSum = 0;

	curandState localState = states[id];

	if (ants[id].path.count == 1){
		for (int i = 0; i < last->neighCount; ++i){
			admissible[i] = last->neigh[i];
		}
		admissibleCount = last->neighCount;

		for (int i = 0; i < admissibleCount; ++i){
			position *tmp = admissible[i];
			probabilities[i] = powf(tmp->pheromone, 4) * powf(tex2D(heuristicsTexture, tmp->i, tmp->j), 2);
			probSum += probabilities[i];
		}
	}

	else {
		position *penultimate = ants[id].path.penultimate();
		for (int i = 0; i < last->neighCount; ++i){
			if (ants[id].path.contains(last->neigh[i])) continue;
			if (last->neigh[i] == penultimate) continue;

			admissible[admissibleCount++] = last->neigh[i];
		}

		for (int i = 0; i < admissibleCount; ++i){
			position *tmp = admissible[i];
			probabilities[i] = powf(tmp->pheromone, 4) * powf(tex2D(heuristicsTexture, tmp->i, tmp->j), 2);
			probSum += probabilities[i];
		}
	}

	double r = curand_uniform_double(&localState) * probSum;
	double acumulatedSum = 0;
	position *next = 0;
	for (int i = 0; i < admissibleCount; ++i){
		acumulatedSum += probabilities[i];
		if (r < acumulatedSum) {
			next = admissible[i];
			break;
		}
	}
	if (!next){
		if (admissibleCount) next = admissible[--admissibleCount];
		else {
			next = ants[id].path[0];
		}
	}

	atomicAdd(&next->antCount, 1);
	ants[id].push_back(next);
	states[id] = localState;
}

/*
 * Kernel koji azurira feromonske tragove
 */
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
