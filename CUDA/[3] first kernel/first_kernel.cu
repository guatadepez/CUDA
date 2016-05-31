#include <stdio.h>

//Utilizar `__global__` es lo mas parecido a declarar un `int main` pero en el device (GPU)
__global__ void device_greetings(void){
    //blockIdx.x representa el id del bloque en el que se encuentra.
    //blockDim.x representa la cantidad total de bloques de la ejecución.
    //threadx.x representa el id del thread en el que se encuentra.
    printf("[DEVICE; BLOCK:%d; THREAD:%d] Hello world!\n",blockIdx.x, threadIdx.x);
}

int main (void){
    //Mostramos un mensaje desde el host (CPU)
    printf("[HOST] Hello world!\n");

    //Aqui es donde se envian llamados paralelos de la funcion `int main` del device:
    //  <<<B,N>>>: donde B es la cantidad de bloques de hilos y N es la cantidad de thread por bloque.
    //En el siguiente ejemplo se ejecutaran 2 bloques con 10 hilos cada uno.
    // Es decir se mostraran 20 mensajes de `Hello world!` desde el device.
    device_greetings<<<2,10>>>();

    //Para mostrar los mensajes del device (GPU), debemos realizar una especie de `barrier`
    // esperando a que todos los bloques y threads terminen. Retornara un mensaje de error en caso de error.
    cudaDeviceSynchronize();

    return 0;
}
