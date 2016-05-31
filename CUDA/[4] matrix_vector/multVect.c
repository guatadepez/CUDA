#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

int *multMatrixVector(int **A, int *B, int *C, int N, int M);
void printVector(int *vector, int N);

int main(int argc, char **argv){
	int N, M;

	scanf("%d %d", &N, &M);

	int **A = (int **)malloc(N * sizeof(int*));
	int *B = (int *)malloc(N * sizeof(int));
	int *C = (int *)malloc(N * sizeof(int));
	
	int i=0;
	int x = 0;
	while(x < N){
		A[i] = (int *)malloc(M * sizeof(int));
		i++;

		int y = 0;
		while(y < M){
			scanf("%d", &A[x][y]);
			y++;
		}
		x++;
	}

	x = 0;
	while(x < N){
		scanf("%d", &B[x]);
		x++;
	}

	x = 0;
	while(x < N){
		scanf("%d", &C[x]);
		x++;
	}

	int *D = multMatrixVector(A, B, C, N, M);
	printVector(D, N);

	return 0;
}

int *multMatrixVector(int **A, int *B, int *C, int N, int M){
	int *D = (int *)malloc(N * sizeof(int));
	int x=0;
	while(x<N){
		int y=0;
		D[x] = 0;
		while(y<M){
			D[x] += A[x][y] * B[y];
			y++;
		}
		D[x] -= C[x];
		x++;
	}
	return D;
}

void printVector(int *vector, int N){
	int x;
	for(x=0; x<N; x++){
		printf("%d  ", vector[x]);
	}
	printf("\n");
}