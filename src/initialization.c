#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "initialization.h"

void set_seed(unsigned int seed){
    srand(seed);
}

static float rand_uniform(float low, float high){
    return low + (high - low) * ((float)rand() / RAND_MAX);
}

void mat_init_weights(Matrix *m, InitMethod method, int n_in, int n_out){
    int size;
    float limit = 0.0f;
    
    if(method == INIT_XAVIER){ //Xavier
        limit = sqrtf(6.0f / (float)(n_in + n_out));
    }else{ //He
        limit = sqrtf(6.0f / (float)n_in);
    }

    size = m->rows * m->stride;
    for(int i=0; i < size; i++){
        m->data[i] = rand_uniform(-limit, limit);
    }
}
