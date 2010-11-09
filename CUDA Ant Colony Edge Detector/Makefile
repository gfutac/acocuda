CUDA_INSTALL_PATH=/usr/local/cuda
CFLAGS= -I. -I$(CUDA_INSTALL_PATH)/include -I"/usr/include/opencv"
LDFLAGS= -L$(CUDA_INSTALL_PATH)/lib64 -lcudart -L"/usr/lib64" -lcv -lhighgui


all:
	nvcc $(CFLAGS) $(LDFLAGS) -arch=sm_12 src/cuda.cu -o Debug/cuda

clean:
	rm -f Debug/*