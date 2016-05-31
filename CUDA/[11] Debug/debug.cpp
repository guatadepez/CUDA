#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <time.h>
#include <math.h>
#include <float.h>

#define PROB_MUTACION 1
#define PROB_CRUCE 0.1
#define PUNISHMENT_F1 30
#define PUNISHMENT_F2_100 5
#define PUNISHMENT_F2_1000 20
#define PUNISHMENT_F2_NONE 50
//#define DEBUG
//#define DEBUG_1

typedef struct {
	float **B;
	float *aptitud;
} Poblacion;

Poblacion init_poblacion(Poblacion poblacion);
double RMSE(float **A, float *B, float *w);
Poblacion tournament_selection(Poblacion poblacion);
void crossover(Poblacion * poblacion);
void mutation_poblacion(Poblacion * poblacion);
float bitwise_mutation_operator(float a);
void fitness(Poblacion * poblacion, float **A, float *B);
int elite(Poblacion poblacion,int elite_index);
Poblacion init_selection(Poblacion selection);
void init();
void init_A();
void init_C();
void init_vector();
float* init_f2();
float ranged_rand(int min, int max);
void display_poblacion(Poblacion poblacion);

int POBLACION;
int N;
int GENERATIONS = 0;
int SUMANDO =0;

float **A;
float *C;
float *VECTOR_FRONTERAS;
int flag;
int flag_2;
int flag_3;
int flag_4;

int main(int argc, char **argv){
	srand(time(NULL));

	std::cin >> N;
	std::cin >> POBLACION;

	Poblacion poblacion;
	Poblacion selection;
	poblacion = init_poblacion(poblacion);
	int elite_index = 0;
	int elite_index_2 = 0;

	init();

	fitness(&poblacion,A,C);

	do{
		//display_poblacion(poblacion);
		GENERATIONS++;		
		selection = tournament_selection(poblacion);		
		crossover(&selection);
		mutation_poblacion(&selection);
		fitness(&selection,A,C);		
		elite_index = elite(poblacion,elite_index);
		elite_index_2 = elite(poblacion,elite_index_2);
		if (poblacion.aptitud[elite_index]>=selection.aptitud[elite_index_2]){
			poblacion.B[POBLACION-1] = selection.B[elite_index];
			poblacion.aptitud[POBLACION-1] = selection.aptitud[elite_index];
		}else{
			poblacion.B[POBLACION-1] = poblacion.B[elite_index];
			poblacion.aptitud[POBLACION-1] = poblacion.aptitud[elite_index];
		}		
		for (int i = 0; i < POBLACION-1; ++i)
		{
			poblacion.B[i]=selection.B[i];
			poblacion.aptitud[i]=selection.aptitud[i];
		}
		//	display_poblacion(poblacion);		
		printf("generation: %d, elite_index_2 numero: %d con valor %f \n", GENERATIONS,elite_index_2,poblacion.aptitud[POBLACION-1]);


	//}while(poblacion.aptitud[elite(poblacion)] > pow(10,-1));	
	}while(GENERATIONS < 100000);	

}

Poblacion init_poblacion(Poblacion poblacion){
	poblacion.B = (float **) malloc (sizeof(Poblacion)*POBLACION);
	for(int i=0;i<POBLACION;i++){
		poblacion.B[i] = (float *) malloc (sizeof(float)*N);
		for(int j=0;j<N;j++)
			poblacion.B[i][j] = ranged_rand(1, 3);
	}
	poblacion.aptitud = (float *) malloc (sizeof(Poblacion)*POBLACION);
	for(int i=0;i<POBLACION;i++)
		poblacion.aptitud[i] = 0;
	return poblacion;
}

void init(){
	init_A();
	init_C();
	init_vector();
}

void init_A(){
	A = (float **) malloc (sizeof(Poblacion)*POBLACION);
	for(int i=0; i<POBLACION; i++){
		A[i] = (float *) malloc (sizeof(float)*N);
		for(int j=0; j<N; j++)
			std::cin >> A[i][j];
	}
}

void init_C(){
	C = (float *) malloc (sizeof(float)*N);
	for(int i=0; i<N; i++)
		std::cin >> C[i];
}

void init_vector(){
	VECTOR_FRONTERAS = (float *) malloc (sizeof(float)*N);
	for(int i=0; i<N; i++)
		std::cin >> VECTOR_FRONTERAS[i];
}

void display_poblacion(Poblacion poblacion){
	printf("\n####### DISPLAY POBLACION #######\n");
	for(int i=0;i<POBLACION;i++){
		printf("Valor del cromosoma |");
		for(int j=0;j<N;j++){
			printf(" %f |",poblacion.B[i][j]);
		}
		printf("\nAptitud :%f \n", poblacion.aptitud[i]);
	}
	printf("################################\n");
}

/****Otras Funciones*****/

float ranged_rand(int min, int max){
    return min + ((float)(max - min) * (rand() / (RAND_MAX + 1.0)));
}

Poblacion tournament_selection(Poblacion poblacion){
	Poblacion selection;
	selection = init_selection(selection);
	//selection = poblacion;
	int cand_a, cand_b;

	for(int i=0;i<POBLACION;i++){
		cand_a = ranged_rand(0,POBLACION);
		cand_b = ranged_rand(0,POBLACION);

		if(poblacion.aptitud[cand_a]<poblacion.aptitud[cand_b]){
			selection.B[i] = poblacion.B[cand_a];
			selection.aptitud[i] = poblacion.aptitud[cand_a];
		} else {
			selection.B[i] = poblacion.B[cand_b];
			selection.aptitud[i] = poblacion.aptitud[cand_b];
		}
	}
	return selection;
}

void crossover(Poblacion * poblacion){

	float aux;
	//	if greater than cero
	if(ranged_rand(-1,1) > 0){		
		for(int i=0;i<POBLACION-1;i+=2){			
			if((double) rand()/(RAND_MAX+1.0) < PROB_CRUCE){
				for(int j=(int)(N/2);j<N;j++){
					aux=poblacion->B[i][j];
					poblacion->B[i][j]=poblacion->B[i+1][j];
					poblacion->B[i+1][j]=aux;
				}
			}			
		}
	} else {
		//	if lower than cero		
		for(int i=0; i<POBLACION-1; i+=2){			
			if((double) rand()/(RAND_MAX+1.0) < PROB_CRUCE){
				for(int j=(int)(N/2), i_aux = 0; j<N; j++, i_aux++){
					aux=poblacion->B[i][i_aux];
					poblacion->B[i][i_aux]=poblacion->B[i+1][j];
					poblacion->B[i+1][j]=aux;
				}
			}			
		}
	}
}

float bitwise_mutation_operator(float a){
	unsigned char *c = reinterpret_cast<unsigned char *>(&a);
	int x = ranged_rand(0,7);
	int quarter = ranged_rand(0,3);

	#ifdef DEBUG
		printf("\na: %f, quarter: %d; c length: %lu; c[%d]: ", a, quarter, sizeof(c), x);
		std::cout << std::bitset<8>(c[x])[x] << std::endl << std::endl;
		for(size_t i=0; i<sizeof a; ++i){
			std::cout << std::bitset<8>(c[i]) << std::endl;
		}
	#endif
	do{
		if (quarter<4){		
    		c[quarter] = c[quarter] ^ (1<<x);
    		flag++;
    	}
    	quarter = ranged_rand(0,3);
	}while((float) *(reinterpret_cast<float *>(c)) >= PTRDIFF_MAX);


	#ifdef DEBUG
	    for(size_t i=0; i<sizeof a; ++i){
			std::cout << std::bitset<8>(c[i]) << std::endl;
		}
		std::cout << std::endl;
	#endif

    return (float) *(reinterpret_cast<float *>(c));
}

void mutation_poblacion(Poblacion * poblacion){

	float *B;
	#ifdef DEBUG
		printf("\n####### MUTACION POBLACION #######\n");
	#endif
	for(int i=0;i<POBLACION;i++){
		B=poblacion->B[i];
		if (ranged_rand(0,1) < PROB_MUTACION){
			int j = (int)ranged_rand(0, N-1);
			B[j] = bitwise_mutation_operator(B[j]);	
		}
		poblacion->B[i] = B;
	}	
	#ifdef DEBUG
		printf("################################\n");
	#endif
}

double RMSE(float **A, float *B, float *w){
	double prod, E;
	E=0;
	for(int i=0;i<POBLACION;i++){
		prod = 0;
		for(int j=0;j<N;j++)
			prod += A[i][j]*w[j];
		E += fabs(prod-B[i]);
	}
	return sqrt(E/POBLACION);
}

void fitness(Poblacion * poblacion, float **A, float *B){

	float F0, F1, F2;
	float * f2 = init_f2();
	for(int i=0;i<POBLACION;i++){		
		poblacion->aptitud[i] = RMSE(A, B, poblacion->B[i]);	  	
	}	
}

float* init_f2(){
	float *f2 = (float *) malloc (sizeof(float)*N);
	for(int i=0;i<N;i++)
		f2[i] = ranged_rand(0, N-1);
	return f2;
}

int elite(Poblacion poblacion,int elite_index){
    int best = elite_index;
    for(int i=0; i<POBLACION; i++){
        if(poblacion.aptitud[best] >= poblacion.aptitud[i])
            best = i;
    }
    return best;
}

Poblacion init_selection(Poblacion selection){
	selection.B = (float **) malloc (sizeof(Poblacion)*POBLACION);
	for(int i=0;i<POBLACION;i++)
		selection.B[i] = (float *) malloc (sizeof(float)*N);
	
	selection.aptitud = (float *) malloc (sizeof(Poblacion)*POBLACION);
	return selection;
}