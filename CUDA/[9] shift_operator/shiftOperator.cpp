#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <time.h>
#include <math.h>
#include <float.h>


#define PROB_MUTACION 1
#define PROB_CRUCE 1
#define PUNISHMENT_F1 30
#define PUNISHMENT_F2_100 5
#define PUNISHMENT_F2_1000 20
#define PUNISHMENT_F2_NONE 50


int POBLACION;
int N;
int GENERATIONS = 1;
int SUMANDO =0;

//#define DEBUG
//#define DEBUG_1

typedef struct {
	float **B;
	float *aptitud;
} Poblacion;

float **A;
float *C;
float *VECTOR_FRONTERAS;

void init();
void init_A();
void init_C();
void init_vector();

Poblacion init_poblacion(Poblacion poblacion);
Poblacion init_selection(Poblacion selection);
void display_poblacion(Poblacion poblacion);
//void display_A();

void crossover(Poblacion * poblacion);
float bitwise_mutation_operator(float a);
void mutation_poblacion(float *B);
Poblacion tournament_selection(Poblacion poblacion);
//int elite(Poblacion poblacion);
int elite(Poblacion poblacion,int elite_index);

float fitness(float **A, float *B, float *w);
double RMSE(float **A, float *B, float *w);
float vector_plus(float *vector);
float* init_f2();
float* init_front();
float ranged_rand(int min, int max);

int main(int argc, char **argv){
	srand(time(NULL));

	std::cin >> N;
	std::cin >> POBLACION;


	Poblacion poblacion;
	poblacion = init_poblacion(poblacion);
	init();	
	
	for(int x=0;x<POBLACION;x++){
		poblacion.aptitud[x] = RMSE(A, C, poblacion.B[x]);
	}
	
	int elite_index = 0;
	do{
		#ifdef DEBUG_1
			display_poblacion(poblacion);
		#endif
		poblacion = tournament_selection(poblacion);		
		crossover(&poblacion);
		elite_index = elite(poblacion,elite_index);
		poblacion.B[POBLACION-1] = poblacion.B[elite_index];
		poblacion.aptitud[POBLACION-1] = poblacion.aptitud[elite_index];



		#ifdef DEBUG
			display_poblacion(poblacion);
			printf("CROMOSOMAS MUTADOS\n");
		#endif

		GENERATIONS++;
		printf("Generacion numero: %d \n",GENERATIONS);
		/*
		for(int m=0;m<POBLACION;m++){
			printf("aptitud de la poblacion: %f en el indice: %d \n\n",poblacion.aptitud[m],m);
		}*/
		printf("aptitud del elite: %f en el indice: %d \n\n",poblacion.aptitud[elite_index],elite_index);

	//}while(poblacion.aptitud[elite(poblacion)] > pow(10,-1));
	}while(GENERATIONS < 5000);		

	int best = elite(poblacion, elite_index);
	#ifdef DEBUG_1
	printf ("\n*************************************\n");
    printf ("*          FIN DEL ALGORITMO        *\n");
    printf ("*************************************\n");
    printf (" - El sujeto mas pulento de la poblacion: [");
    for(int i=0; i<POBLACION-1; i++){
    	if(i == POBLACION-2)
    		printf("%f]\n", poblacion.B[best][i]);
    	else
    		printf("%f-", poblacion.B[best][i]);
    }
    printf (" - Su aptitud/fenotipo es %.5f\n", poblacion.aptitud[best]);
    printf (" - Es la generacion numero %i\n", GENERATIONS);
    printf ("\n sumando :%d\n", SUMANDO);
    printf ("*************************************\n");
    #endif
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

void display_A(){	
	for(int i=0; i<POBLACION; i++){		
		for(int j=0; j<N; j++){
			printf("%f ",A[i][j] );			
		}
		printf("\n" );
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

Poblacion init_selection(Poblacion selection){
	selection.B = (float **) malloc (sizeof(Poblacion)*POBLACION);
	for(int i=0;i<POBLACION;i++)
		selection.B[i] = (float *) malloc (sizeof(float)*N);
	
	selection.aptitud = (float *) malloc (sizeof(Poblacion)*POBLACION);
	return selection;
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

void mutation_poblacion(float *B){
	#ifdef DEBUG
		printf("\n####### MUTACION POBLACION #######\n");
	#endif
	if ((double)rand()/(RAND_MAX+1.0) < PROB_MUTACION){
		int j = (int)ranged_rand(0, N-1);
		B[j] = bitwise_mutation_operator(B[j]);	
	}
	#ifdef DEBUG
		printf("################################\n");
	#endif
}

/*  Debemos obtener una forma de cambiar un bit no tan drasticamente, ya que si no lo hacemos bien,
	los numeros pueden llegar a cambiar demasiado.
	//x = a/(rand()%a);
    Obtenido de esta pagina: http://www.cprogramming.com/tutorial/bitwise_operators.html
	n_use = in_use ^ 1<<car_num;
*/
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

double RMSE(float **A, float *B, float *w){
	double prod, E;
	for(int i=0;i<POBLACION;i++){
		prod = 0;
		for(int j=0;j<N;j++)
			prod += A[i][j]*w[j];
		E += fabs(pow(prod-B[i],2));
	}
	return sqrt(E/POBLACION);
}

float vector_plus(float *vector){
	float total = 0;
	for(int i=0; i<N; i++)
		total += vector[i];
	return total;
}

/*
*	TODO: preguntar largo de vector frontera.
*/
float* init_front(){
	float *front = (float *) malloc (sizeof(float)*N);
	for(int i=0;i<N;i++)
		front[i] = pow(2, i);
	return front;
}

/*
*	front viene ordenado.
*/
float fitness(float **A, float *B, float *w){
	float F0, F1, F2;
	float * f2 = init_f2();
	
	//	F0: primera condicion RMSE
	F0 = RMSE(A, B, w);

	//	F1: Condiciones originales de fitness implementada en el paper de Arturo Benson.
	if (POBLACION == 289 || POBLACION == 1089){
		if(vector_plus(w) < vector_plus(B))
			F1 = 0;
		else
			F1 = PUNISHMENT_F1;
	}

	//F2: condiciones.
	for(int node=0, i=0; node<POBLACION; node++, i++){
		if(std::find(VECTOR_FRONTERAS, VECTOR_FRONTERAS+N, node)){
			if(w[node] == 0)
				f2[i] = 0;
			else if(w[node] <= 100)
				f2[i] = PUNISHMENT_F2_100;
			else if(w[node] <= 1000)
				f2[i] = PUNISHMENT_F2_1000;
			else
				f2[i] = PUNISHMENT_F2_NONE;
		}
		if(i>N)
			break;
	}
	F2 = vector_plus(f2);

	if (POBLACION == 289 || POBLACION == 1089)
  		//return F0+F1+F2;
  		return F0;
  	else
  		return F0;
}

float* init_f2(){
	float *f2 = (float *) malloc (sizeof(float)*N);
	for(int i=0;i<N;i++)
		f2[i] = ranged_rand(0, N-1);
	return f2;
}

/*void cruzamiento_poblacion(){
dataLength= 8*sizeof(Datatype);
realLength=ceil(solutionLength/dataLength);
word=crossoverPoint/dataLength;
wordPoint=crossoverPoint%dataLength;
restW P=dataLength-wordPoint;
snew1[word] = ((si[word]>>restWP)<<restWP)|((s2[word]<<wordPoint)>>wordPoint);
snew2[word] = ((s2[word]>>restWP)<<restWP)|((s1[word]<<wordPoint)>>wordPoint);

got it from = https://www.lri.fr/~hansen/proceedings/2011/GECCO/companion/p439.pdf
*/
//void crossover(Poblacion poblacion, int individual_a, int individual_b){
void crossover(Poblacion * poblacion){
	float aux;
	//	if one
	if(ranged_rand(-1,1)>0){
		for(int i=0;i<POBLACION-1;i+=2){
			
			if((double) rand()/(RAND_MAX+1.0) < PROB_CRUCE){
				for(int j=(int)(N/2);j<N;j++){
					aux=poblacion->B[i][j];
					poblacion->B[i][j]=poblacion->B[i+1][j];
					poblacion->B[i+1][j]=aux;
				}
			}
			mutation_poblacion(poblacion->B[i]);
			/*			
			Donde A es la matriz que se obtiene de los datos, C es el Vector que se resta a la ecuación y poblacion-B es el
			Vector que se trabaja en el AG.
			*/
			//poblacion->aptitud[i] = fitness(A, C, poblacion->B[i]);
			poblacion->aptitud[i] =RMSE(A, C, poblacion->B[i]);

		}
	} else {
		//	if zero
		for(int i=0; i<POBLACION-1; i+=2){
			
			if((double) rand()/(RAND_MAX+1.0) < PROB_CRUCE){
				for(int j=(int)(N/2), i_aux = 0; j<N; j++, i_aux++){
					aux=poblacion->B[i][i_aux];
					poblacion->B[i][i_aux]=poblacion->B[i+1][j];
					poblacion->B[i+1][j]=aux;
				}
			}
			mutation_poblacion(poblacion->B[i]);
			/*
			Donde A es la matriz que se obtiene de los datos, C es el Vector que se resta a la ecuación y poblacion-B es el
			Vector que se trabaja en el AG.
			*/
			//poblacion->aptitud[i] = fitness(A, C, poblacion->B[i]);
			poblacion->aptitud[i] =RMSE(A, C, poblacion->B[i]);
		}
	}
}

Poblacion tournament_selection(Poblacion poblacion){
	Poblacion selection;
	//selection = init_selection(selection);
	selection = poblacion;
	int cand_a, cand_b;

	for(int i=0;i<POBLACION;i++){
		cand_a = ranged_rand(0,POBLACION);
		cand_b = ranged_rand(0,POBLACION);

		if(poblacion.aptitud[cand_a]<poblacion.aptitud[cand_b]){
			selection.B[cand_b] = poblacion.B[cand_a];
			selection.aptitud[cand_b] = poblacion.aptitud[cand_a];
		} else {
			selection.B[cand_a] = poblacion.B[cand_b];
			selection.aptitud[cand_a] = poblacion.aptitud[cand_b];
		}
	}
	return selection;
}

int elite(Poblacion poblacion,int elite_index){
    int best = elite_index;
    for(int i=0; i<POBLACION; i++){
        if(poblacion.aptitud[best] > poblacion.aptitud[i])
            best = i;
    }
    return best;
}

float ranged_rand(int min, int max){
    return min + ((float)(max - min) * (rand() / (RAND_MAX + 1.0)));
}