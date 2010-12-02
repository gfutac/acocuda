#include "position.h"

struct vector{
	position *visitedPositions[40];
	int pos;
	int len;
	int count;

	vector() { }

	__device__ vector(int len) : len(len) {
		pos = 0;
		count = 0;
	}

	__device__ void push_back(position *elem){
		pos = count++ % len;
		visitedPositions[pos] = elem;
	}

	__device__ position* last(){
		return visitedPositions[pos];
	}

	__device__ position* penultimate(){
		int position = pos - 1;
		if (position < 0) position = len - 1;
		return visitedPositions[position];
	}

	__device__ position* operator[](int index){
		return visitedPositions[index];
	}

	__device__ bool contains(position* other){
		for (int i = count - 1; i >=  count - 1 - len; --i){
			if (visitedPositions[i % len] == other) return true;
		}
		return false;
	}
};
