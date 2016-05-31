#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <time.h>
#include <math.h>
#include <float.h>
#include <limits.h>


/*************************************************************************************************
**************************************************************************************************
									Declaración de Macros
**************************************************************************************************
***************************************************************************************************/
#define PROB_MUTACION 1
#define PROB_CRUCE 1
#define PUNISHMENT_F1 3
#define PUNISHMENT_F2_100 5

typedef struct {
	int *genotype;
	int *aptitud;
} Cromosome;

int POBLACION;
int N;
int objetivo;

/*************************************************************************************************
**************************************************************************************************
									Declaración de funciones
**************************************************************************************************
***************************************************************************************************/


Cromosome init_crom(Cromosome poblacion);
Cromosome init_crom_sel(Cromosome poblacion);
int ranged_rand(int min, int max);

void show_crom(Cromosome poblacion);

int min_function(int A);
void fitness_function(Cromosome * poblacion);
Cromosome tournament_selection_2(Cromosome poblacion);
int elite(Cromosome poblacion,int elite_index);
void mutation_poblacion(Cromosome * poblacion);
int bitwise_mutation_operator(int a);
void crossover(Cromosome * poblacion);

/*************************************************************************************************
**************************************************************************************************
											Main
**************************************************************************************************
***************************************************************************************************/
int main(){
	POBLACION = 50;
	objetivo = INT_MAX;
	std::cout << std::fixed << "int max: " << INT_MAX << std::endl;
	Cromosome poblacion;
	Cromosome seleccion;	
	poblacion = init_crom(poblacion);
	seleccion = init_crom_sel(seleccion);
	fitness_function(&poblacion);	
	int GENERATIONS;
	int elite_index=0;
	int elite_index_2=0;
	int MINIMO_GLOBAL=0;
	MINIMO_GLOBAL = 0;
	do{
		GENERATIONS++;		
		seleccion = tournament_selection_2(poblacion);
		crossover(&seleccion);
		mutation_poblacion(&seleccion);
		fitness_function(&seleccion);		
		elite_index = elite(poblacion,elite_index);			
		elite_index_2 = elite(seleccion,elite_index_2);		
		
		if(poblacion.aptitud[elite_index] > seleccion.aptitud[elite_index_2]){
			poblacion.genotype[POBLACION-1] = seleccion.genotype[elite_index_2];
			poblacion.aptitud[POBLACION-1] = seleccion.aptitud[elite_index_2];
			MINIMO_GLOBAL = seleccion.aptitud[elite_index_2];
		}else{
			//poblacion.genotype[POBLACION-1] = poblacion.genotype[elite_index];
			//poblacion.aptitud[POBLACION-1] = poblacion.aptitud[elite_index];
			MINIMO_GLOBAL = poblacion.aptitud[POBLACION-1];
		}
		for (int i = 0; i < POBLACION-1; ++i)
		{
			poblacion.genotype[i]=seleccion.genotype[i];
			poblacion.aptitud[i]=seleccion.aptitud[i];

		}
		//show_crom(poblacion);		
		/*
		std::cout << std::fixed << "----------------- "  << std::endl;		
		std::cout << std::fixed << "elite_index: " << elite_index << std::endl;
		std::cout << std::fixed << "pob_geno: " << poblacion.genotype[elite_index] << std::endl;
		std::cout << std::fixed << "pob_aptitud: " << poblacion.aptitud[elite_index] << std::endl;		
		std::cout << std::fixed << "elite_index_2: " << elite_index_2 << std::endl;
		std::cout << std::fixed << "sel_geno: " << seleccion.genotype[elite_index] << std::endl;
		std::cout << std::fixed << "sel_aptitud: " << seleccion.aptitud[elite_index_2] << std::endl;
		std::cout << std::fixed << "generacion: " << GENERATIONS << std::endl;
		std::cout << std::fixed << "aptitud del mejor: " << MINIMO_GLOBAL << std::endl;
		std::cout << std::fixed << "----------------- "  << std::endl;
		*/
		std::cout << std::fixed << "minimo obtenido: " << MINIMO_GLOBAL << std::endl;
		std::cout << std::fixed << poblacion.genotype[POBLACION-1]  << std::endl;
	}while(abs(INT_MAX - seleccion.genotype[elite_index_2]) > 1);


	std::cout << std::fixed << "----------------- "  << std::endl;
	std::cout << std::fixed << "elite_index: " << elite_index << std::endl;
	std::cout << std::fixed << "pob_geno: " << poblacion.genotype[elite_index] << std::endl;
	std::cout << std::fixed << "pob_aptitud: " << poblacion.aptitud[elite_index] << std::endl;		
	std::cout << std::fixed << "----------------- "  << std::endl;
	std::cout << std::fixed << "elite_index_2: " << elite_index_2 << std::endl;			
	std::cout << std::fixed << "sel_geno: " << seleccion.genotype[elite_index] << std::endl;
	std::cout << std::fixed << "sel_aptitud: " << seleccion.aptitud[elite_index_2] << std::endl;
	std::cout << std::fixed << "----------------- "  << std::endl;
	std::cout << std::fixed << "Resultados finales *********** "  << std::endl;
	std::cout << std::fixed << "generacion: " << GENERATIONS << std::endl;
	std::cout << std::fixed << "aptitud del mejor: " << poblacion.aptitud[POBLACION-1] << std::endl;
	std::cout << std::fixed << "minimo obtenido: " << MINIMO_GLOBAL << std::endl;
	std::cout << std::fixed << "objetivo: " << objetivo << std::endl;
	std::cout << std::fixed << "minimo del genotipo: " << poblacion.genotype[POBLACION-1] << std::endl;
	int final = 0;
	final = objetivo - abs(seleccion.genotype[elite_index_2]);
	std::cout << std::fixed << "diferencia de partes: " << final << std::endl;	
	std::cout << std::fixed << "----------------- "  << std::endl;

}


/*************************************************************************************************
**************************************************************************************************
									Funciones de iniciación
**************************************************************************************************
***************************************************************************************************/

Cromosome init_crom(Cromosome poblacion){
	poblacion.genotype = (int *) malloc(sizeof(int)*POBLACION);
	for (int i = 0; i < POBLACION; ++i){
		poblacion.genotype[i] = ranged_rand(0,INT_MAX);
	}
	poblacion.aptitud = (int *) malloc(sizeof(int)*POBLACION);
	for (int i = 0; i < POBLACION; ++i){
		//Cromosome.aptitud[i] = aptitud();
		poblacion.aptitud[i] = 0;
	}
	return poblacion;
}
Cromosome init_crom_sel(Cromosome poblacion){
	poblacion.genotype = (int *) malloc(sizeof(int)*POBLACION);
	for (int i = 0; i < POBLACION; ++i){
		poblacion.genotype[i] = 0;
	}
	poblacion.aptitud = (int *) malloc(sizeof(int)*POBLACION);
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
	int score;
	for (int i = 0; i < POBLACION; ++i)
	{
		score = min_function(poblacion->genotype[i]);		
		poblacion->aptitud[i] = score;
	}
}

int min_function(int A){
	int result;
	//std::cout << std::fixed << objetivo << std::endl;	
	result = INT_MAX - abs(A);	
	return result;
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

	int B;
	for(int i=0;i<POBLACION;i++){
		B = poblacion->genotype[i];
		if (ranged_rand(0,1) < PROB_MUTACION){			
			B = bitwise_mutation_operator(B);	
			poblacion->genotype[i] = B;
		}
		
	}	
}
/*
int bitwise_mutation_operator(int a){
	unsigned char *c = reinterpret_cast<unsigned char *>(&a);
	int x = ranged_rand(0,7);
	int quarter = ranged_rand(0,4);

		do{
			if (quarter<4){		
	    		c[quarter] = c[quarter] ^ (1<<x);
	    	}
	    	quarter = ranged_rand(0,4);
	    	
		}while((int) *(reinterpret_cast<int *>(c)) >= INT_MAX);

    return (int) *(reinterpret_cast<int *>(c));
}
*/

int bitwise_mutation_operator(int a){
	int c = a;
	int x = ranged_rand(0,31);
	int piv;
		do{
			//std::cout << std::fixed << "x: " << x << std::endl;
			c = c  ^ (1<<x);			
		}while( c >= INT_MAX);

    return c;
}

/*   Cruzamiento   */

void crossover(Cromosome * poblacion){

	int dataLength = INT_MAX;
	int realLength = 16;
	int crossoverPoint = 16;
	//int Mask = 65535;
	int wordPoint = 16;

	for (int i = 0; i < POBLACION-1; i=i+2)
	{
		int piv1 = poblacion->genotype[i];
		int piv2 = poblacion->genotype[i+1];		
		int restWP = dataLength-realLength;
		int snew1 = ((piv1>>crossoverPoint)<<crossoverPoint)|(((piv2<<wordPoint))>>wordPoint);	
		int snew2 = ((piv2>>crossoverPoint)<<crossoverPoint)|(((piv1<<wordPoint))>>wordPoint);

		poblacion->genotype[i] = snew1;
		poblacion->genotype[i+1] = snew2;

	}
	


}


/*************************************************************************************************
**************************************************************************************************
									Otras Funciones
**************************************************************************************************
***************************************************************************************************/

int ranged_rand(int min, int max){
    return min + ((int)(max - min) * (rand() / (RAND_MAX + 1.0)));
}

