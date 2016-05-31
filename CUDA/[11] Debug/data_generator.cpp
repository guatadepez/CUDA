#include <iostream>
#include <cstdlib>
#include <cstdio>

float ranged_rand(int min, int max);

int main(int argc, char **argv) {
	srand(time(NULL));

	if (argc < 3) {
        std::cerr << "[Usage: " << argv[0] << " { N } { PROBABILIDAD_MATRIZ } ]" << std::endl;
        return 1;
    }

    int N = atoi(argv[1]);
    printf("%d\n%d\n\n", N, N);

    float PROBABILIDAD_MATRIZ = atof(argv[2]);

    //printf("N = %d, PROBABILIDAD_MATRIZ = %f\n", N, PROBABILIDAD_MATRIZ);

	float random;
	for(int i=0; i<N; i++){
		for(int j=0; j<N; j++){
			random = ranged_rand(0, 1);
			printf("%f ", random < PROBABILIDAD_MATRIZ ? 0.0f : random);
		}
		printf("\n");
	}

	printf("\n");

	for(int i=0; i<N; i++){
		for(int j=0; j<1; j++)
			printf("%f ", ranged_rand(0, 1));
		printf("\n");
	}

	printf("\n");

	for(int i=0; i<N; i++){
		for(int j=0; j<1; j++)
			printf("%f ", ranged_rand(0, 1));
		printf("\n");
	}

	printf("\n");

	for(int i=0; i<N; i++){
		for(int j=0; j<1; j++)
			printf("%d,%d", (int)ranged_rand(0, N), (int)ranged_rand(0, N));
		printf("\n");
	}

}

float ranged_rand(int min, int max){
    return min + ((float)(max - min) * (rand() / (RAND_MAX + 1.0)));
}