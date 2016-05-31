#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <time.h>
#include <math.h>
#include <float.h>


/*************************************************************************************************
**************************************************************************************************
									Declaración de Macros
**************************************************************************************************
***************************************************************************************************/
#define PROB_MUTACION 1
#define PROB_CRUCE 1
#define PUNISHMENT_F1 3
#define PUNISHMENT_F2_100 5
//#define DEBUG

typedef struct {
	long double *genotype;
	long double *aptitud;
} Cromosome;

int POBLACION;
int N;
long double objetivo;
int best;

/*************************************************************************************************
**************************************************************************************************
									Declaración de funciones
**************************************************************************************************
***************************************************************************************************/

long double ranged_rand(float min, long double max);
Cromosome init_crom(Cromosome poblacion);
Cromosome init_crom_sel(Cromosome poblacion);
void show_crom(Cromosome poblacion);
void fitness_function(Cromosome * poblacion);
long double min_function(long double A);
Cromosome tournament_selection(Cromosome poblacion);
Cromosome tournament_selection_2(Cromosome poblacion);
int elite(Cromosome poblacion,int elite_index);
void mutation_poblacion(Cromosome * poblacion);
long double bitwise_mutation_operator(long double a);
void crossover(Cromosome * poblacion);


/*************************************************************************************************
**************************************************************************************************
											Main
**************************************************************************************************
***************************************************************************************************/
int main(){
	POBLACION = 4;
	objetivo = DBL_MAX;
	best = 0;
	Cromosome poblacion;
	Cromosome seleccion;	
	poblacion = init_crom(poblacion);
	fitness_function(&poblacion);
	seleccion = init_crom_sel(seleccion);		
	seleccion = tournament_selection_2(poblacion);
	best = elite(seleccion,best);	
	//mutation_poblacion(&seleccion);
	crossover(&seleccion);
	//show_crom(seleccion);
	int index_elite;
	float apt_index;

}


/*************************************************************************************************
**************************************************************************************************
									Funciones de iniciación
**************************************************************************************************
***************************************************************************************************/

Cromosome init_crom(Cromosome poblacion){
	poblacion.genotype = (long double *) malloc(sizeof(long double)*POBLACION);
	for (int i = 0; i < POBLACION; ++i){
		poblacion.genotype[i] = ranged_rand(0,DBL_MAX);
	}
	poblacion.aptitud = (long double *) malloc(sizeof(long double)*POBLACION);
	for (int i = 0; i < POBLACION; ++i){
		//Cromosome.aptitud[i] = aptitud();
		poblacion.aptitud[i] = 0;
	}
	return poblacion;
}
Cromosome init_crom_sel(Cromosome poblacion){
	poblacion.genotype = (long double *) malloc(sizeof(long double)*POBLACION);
	for (int i = 0; i < POBLACION; ++i){
		poblacion.genotype[i] = 0;
	}
	poblacion.aptitud = (long double *) malloc(sizeof(long double)*POBLACION);
	for (int i = 0; i < POBLACION; ++i){		
		poblacion.aptitud[i] = 0;
	}
	return poblacion;
}


/*************************************************************************************************
**************************************************************************************************
									Funciones de Muestra
**************************************************************************************************
***************************************************************************************************/

void show_crom(Cromosome poblacion){
	for (int i = 0; i < POBLACION; ++i)
	{		
		std::cout << std::fixed << "genotipo:" << poblacion.genotype[i] << std::endl;
		std::cout << std::fixed << "aptitud: " << poblacion.aptitud[i] << std::endl;
		
	}
}


/*************************************************************************************************
**************************************************************************************************
									Funciones del AG
**************************************************************************************************
***************************************************************************************************/

/****Función FItness****/

void fitness_function(Cromosome * poblacion){
	long double score;
	for (int i = 0; i < POBLACION; ++i)
	{
		score = min_function(poblacion->genotype[i]);
		if (score < 0)
			poblacion->aptitud[i] = score + FLT_MAX;
		else
			poblacion->aptitud[i] = score;
	}
}

long double min_function(long double A){
	long double result;
	//std::cout << std::fixed << objetivo << std::endl;
	result = (objetivo - A);
	//std::cout << std::fixed << result << std::endl;
	
	return result;
}

/****Seleccion****/

Cromosome tournament_selection(Cromosome poblacion){
	Cromosome seleccion;
	seleccion = init_crom_sel(seleccion);	
	int count;
	count = 0;
	for(int i=0;i<POBLACION-1;i=i+2){
		if(poblacion.aptitud[i]>poblacion.aptitud[i+1]){
			seleccion.genotype[count] = poblacion.genotype[i];
			seleccion.aptitud[count] = poblacion.aptitud[i];
		}else{
			seleccion.genotype[count] = poblacion.genotype[i+1];
			seleccion.aptitud[count] = poblacion.aptitud[i+1];
		}
		std::cout << std::fixed << "i: "<< i << "count :" << count << std::endl;
		count++;		
	}
	
	return seleccion;
}

/*   selection 2 (para ver diferencias de comportamiento) */

Cromosome tournament_selection_2(Cromosome poblacion){
	Cromosome selection;
	selection = init_crom_sel(selection);	
	int cand_a, cand_b;

	for(int i=0;i<POBLACION;i++){
		cand_a = ranged_rand(0,POBLACION);
		cand_b = ranged_rand(0,POBLACION);

		if(poblacion.aptitud[cand_a]<poblacion.aptitud[cand_b]){
			selection.genotype[i] = poblacion.genotype[cand_a];
			selection.aptitud[i] = poblacion.aptitud[cand_a];
		} else {
			selection.genotype[i] = poblacion.genotype[cand_b];
			selection.aptitud[i] = poblacion.aptitud[cand_b];
		}
	}
	return selection;
}

/*   Elite   */

int elite(Cromosome poblacion,int elite_index){
    int best = elite_index;
    for(int i=0; i<POBLACION; i++){
        if(poblacion.aptitud[best] > poblacion.aptitud[i])
            best = i;
    }
    return best;
}

/* Mutacion */

void mutation_poblacion(Cromosome * poblacion){

	long double B;
	for(int i=0;i<POBLACION;i++){
		B = poblacion->genotype[i];
		if (ranged_rand(0,1) < PROB_MUTACION){			
			B = bitwise_mutation_operator(B);	
			poblacion->genotype[i] = B;
		}
		
	}	
}

long double bitwise_mutation_operator(long double a){
	unsigned char *c = reinterpret_cast<unsigned char *>(&a);
	int x = ranged_rand(0,7);
	int quarter = ranged_rand(0,4);

		do{
			if (quarter<4){		
	    		c[quarter] = c[quarter] ^ (1<<x);
	    	}
	    	quarter = ranged_rand(0,4);
	    	
		}while((double) *(reinterpret_cast<double *>(c)) >= UINTMAX_MAX);

    return (long double) *(reinterpret_cast<long double *>(c));
}


/*   Cruzamiento   */

/*
	long double dataLength = 32;
	long double realLength = 32;
	long double crossoverPoint = 1;
	long double Mask = 255;
	long double word = crossoverPoint/realLength;
	long double wordPoint = crossoverPoint%realLength;
	long double restWP = dataLength-realLength;
	long double snew1 = ((a>>crossoverPoint)<<crossoverPoint)|(((b<<wordPoint)&Mask)>>wordPoint);	
	long double snew2 = ((b>>crossoverPoint)<<crossoverPoint)|(((a<<wordPoint)&Mask)>>wordPoint);

*/
/*	
void bitwise_crossover_operator(long double * a, long double * b){

	unsigned char snew1;
	unsigned char snew2;
	unsigned char *piv1 = reinterpret_cast<unsigned char *>(&a);
	unsigned char *piv2 = reinterpret_cast<unsigned char *>(&b);
	int dataLength = 32;
	int realLength = 32;
	int crossoverPoint = 1;
	int Mask = 255;
	int word = crossoverPoint/realLength;
	int wordPoint = crossoverPoint%realLength;
	int restWP = dataLength-realLength;
	snew1 = ((piv1>>crossoverPoint)<<crossoverPoint)|(((piv2<<wordPoint)&Mask)>>wordPoint);	
	snew2 = ((piv2>>crossoverPoint)<<crossoverPoint)|(((piv1<<wordPoint)&Mask)>>wordPoint);

	a = (long double) *(reinterpret_cast<long double *>(snew1));
	b = (long double) *(reinterpret_cast<long double *>(snew2));
	
}
*/

void crossover(Cromosome * poblacion){

	for(int i=0;i<POBLACION-1;i=i+2){
		if (ranged_rand(0,1) < PROB_CRUCE){			
			unsigned char *piv1 = reinterpret_cast<unsigned char *>(&poblacion->genotype[i]);
			unsigned char *piv2 = reinterpret_cast<unsigned char *>(&poblacion->genotype[i+1]);
			unsigned char *snew1;
			snew1 = ((piv1>>piv1)<<piv1);	


			std::cout << std::fixed << "genotipo:" << piv1 << std::endl;
			std::cout << std::fixed << "genotipo:" << piv2 << std::endl;
		}		
	}	
}


/*************************************************************************************************
**************************************************************************************************
									Otras Funciones
**************************************************************************************************
***************************************************************************************************/

long double ranged_rand(float min, long double max){
    return min + ((long double)(max - min) * (rand() / (RAND_MAX + 1.0)));
}

