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

static float rand_normal(float mean, float std){
    static int has_spare = 0;
    static float spare;
    
    if(has_spare){
        has_spare = 0;
        return mean + std * spare;
    }
    
    float u, v, s, scale;

    do{
        u = rand_uniform(-1.0f, 1.0f);
        v = rand_uniform(-1.0f, 1.0f);
        s = (u * u) + (v * v);
    }while(s >= 1.0f || s == 0.0f);
    
    scale = sqrtf(-2.0f * logf(s) / s);
    spare = v * scale;
    has_spare = 1;
    
    return mean + std * (u * scale);
}

void mat_init_weights(Matrix *m, InitMethod method, int n_in, int n_out){
    int size, is_normal = 0;
    float std = 0.0f, limit = 0.0f;
    
    switch(method){
        case INIT_XAVIER:{
            std = sqrtf(2.0f / (float)(n_in + n_out));
            is_normal = 1;
            break;
        }
        case INIT_XAVIER_UNIFORM:{
            limit = sqrtf(6.0f / (float)(n_in + n_out));
            is_normal = 0;
            break;
        }
        case INIT_HE:{
            std = sqrtf(2.0f / (float)n_in);
            is_normal = 1;
            break;
        }
        case INIT_HE_UNIFORM:{
            limit = sqrtf(6.0f / (float)n_in);
            is_normal = 0;
            break;
        }
    }

    size = m->rows * m->stride;
    for(int i=0; i < size; i++){
        if(is_normal){
            m->data[i] = rand_normal(0.0f, std);
        }else{
            m->data[i] = rand_uniform(-limit, limit);
        }
    }
}
