#include "myVector.h"

struct ant{
	myVector path;
	int startx, starty;

	__host__ __device__ ant() { }

	__host__ __device__ ant(int r){
		path = myVector(r);
	}

	__device__ void push_back(pixel *p){
		path.push_back(p);
	}

	__host__ void setStartPosition(int i, int j){
		startx = i;
		starty = j;
	}
};
