#include "pixel.h"

struct myVector{
	pixel *pixels[40];
	int pos;
	int len;
	int count;

	__host__ myVector() {}

	__host__ __device__ myVector(int _len){
		len = _len;
		pos = 0;
		count = 0;
	}

	__device__ void push_back(pixel *elem){
		pos = count % len;
		count += 1;
		pixels[pos] = elem;
	}

	__device__ pixel* last(){
		return pixels[pos];
	}

	__device__ pixel* penultimate(){
		int position = pos - 1;
		if (position < 0) position = len - 1;
		return pixels[position];
	}

	__device__ pixel* operator[](int index){
		return pixels[index];
	}

	__device__ bool contains(pixel* other){
		for (int i = 0; i < len; ++i){
			if (pixels[i] == other) return true;
		}
		return false;
	}

	__host__ __device__ int size(){
		return len;
	}

	__host__ __device__ int getCount(){
		return count;
	}
};
