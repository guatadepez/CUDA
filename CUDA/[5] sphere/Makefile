all:
	nvcc -O3 sphere.cu -o sphere.out -arch=compute_30 -I  /usr/local/cuda/samples/common/inc -L  /usr/local/cuda/samples/common/lib --compiler-options -Wall -lm
clean: rm -rf sphere.out
