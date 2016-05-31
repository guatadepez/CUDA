#include <stdio.h>

typedef struct {
	char * str;
	unsigned int fitness;
} hello_world;

int main(){
	srand(unsigned(time(NULL)));
}

void init_population(hello_world *population, ga_vector &buffer ) {
    int tsize = GA_TARGET.size();

    for (int i=0; i<GA_POPSIZE; i++) {
        ga_struct citizen;
        
        citizen.fitness = 0;
        citizen.str.erase();
                                                                                                                                                               
        for (int j=0; j<tsize; j++)
            citizen.str += (rand() % 90) + 32; 

        population.push_back(citizen);
    }   

    buffer.resize(GA_POPSIZE);
}
