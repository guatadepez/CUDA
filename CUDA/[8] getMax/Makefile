all:
	nvcc -O3 max.cu -o max.out -arch=compute_30 -I  /usr/local/cuda/samples/common/inc -L  /usr/local/cuda/samples/common/lib --compiler-options -Wall
clean: rm -rf max.out
