#include <stdio.h>
#include <cuda.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#define BLOCKSIZE 256
#define WARPSIZE 32

int N=1024;

typedef struct {
    float num;
} Dato;

__inline__ __device__
float warpAllReduceCompare(float val) {
    for (unsigned int mask = WARPSIZE/2; mask > 0; mask /= 2){
        val = fmax(val,__shfl_down(val, mask, WARPSIZE));
    }
    return val;
}

__inline__ __device__
float blockReduceCompare(float val) {
    static __shared__ float shared[WARPSIZE]; // Shared mem for 32 partial sums
    int lane = threadIdx.x % WARPSIZE;
    int wid = threadIdx.x / WARPSIZE;

    val = warpAllReduceCompare(val);     // Each warp performs partial reduction

    if (lane==0) shared[wid]=val; // Write reduced value to shared memory
        __syncthreads();              // Wait for all partial reductions

    //read from shared memory only if that warp existed
    val = (threadIdx.x < blockDim.x / WARPSIZE) ? shared[lane] : 0;

    if (wid==0) val = warpAllReduceCompare(val); //Final reduce within first warp
        return val;
}


__device__ float atomicMaxf(float* address, float val) {
    int *address_as_int =(int*)address;
    int old = *address_as_int, assumed;
    while (val > __int_as_float(old)) {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
        __float_as_int(val));
    }
    return __int_as_float(old);
}

__global__
void eliteKernel(Dato * device_datos){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float MAX = -1.0f;
    MAX = MAX < device_datos[i].num ? device_datos[i].num : MAX;

    MAX = blockReduceCompare(MAX);
    if(threadIdx.x==0){
        atomicMaxf(&device_datos[0].num, MAX);
    }
}

void fill(Dato *host_datos);
float max(Dato * host_datos);

int main(int argc, char ** argv){
    srand(time(NULL));
    StopWatchInterface *cpu_timer;
    sdkCreateTimer(&cpu_timer);
    sdkResetTimer(&cpu_timer);

    StopWatchInterface *gpu_timer;
    sdkCreateTimer(&gpu_timer);
    sdkResetTimer(&gpu_timer);

    StopWatchInterface *timer;
    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);

    if(argc==3)
        N = atoi(argv[1]);
    Dato *host_datos;
    Dato *device_datos;

    host_datos = (Dato *) malloc (sizeof(Dato)*N);
    cudaMalloc((void**)&device_datos, sizeof(Dato)*N);

    sdkStartTimer(&timer);
    fill(host_datos);
    sdkStopTimer(&timer);
    printf("fill time: %f\n", sdkGetTimerValue(&timer)/1000.0f);

    float FINALMAX = -1;
    if(atoi(argv[2])==1){
        sdkStartTimer(&cpu_timer);
        FINALMAX = max(host_datos);
        sdkStopTimer(&cpu_timer);
        printf("max time: %f\n", sdkGetTimerValue(&cpu_timer)/1000.0f);
    }else{
        int GRIDSIZE = (N+BLOCKSIZE-1)/BLOCKSIZE;
        dim3 block(BLOCKSIZE, 1, 1);
        dim3 grid(GRIDSIZE, 1, 1);

        cudaMemcpy(device_datos, host_datos, sizeof(Dato)*N, cudaMemcpyHostToDevice);
        sdkStartTimer(&gpu_timer);
        eliteKernel<<<grid,block>>>(device_datos);
        cudaDeviceSynchronize();
        cudaMemcpy(&FINALMAX, device_datos, sizeof(Dato), cudaMemcpyDeviceToHost);
        sdkStopTimer(&gpu_timer);

        printf("max time: %f\n", sdkGetTimerValue(&gpu_timer)/1000.0f);
    }
    printf("MAX = %f\n",FINALMAX);

    return 0;
}

void fill(Dato *host_datos){
    int i;
    for(i=0;i<N;i++){
        host_datos[i].num=(float)(rand()/(RAND_MAX/(99.0-0.1)));
    }
}

float max(Dato * host_datos){
    int i;
    float max = -1;
    for (i=0;i<N;i++){
        if(host_datos[i].num>max)
            max=host_datos[i].num;
    }
    return max;
}
