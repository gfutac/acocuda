#include "myVector.h"

struct ant{
	vector path;

	__device__ ant(int r){
		path = vector(r);
	}

	__device__ void push_back(position *p){
		path.push_back(p);
	}
};
