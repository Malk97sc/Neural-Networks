#ifndef INITIALIZATION_H
#define INITIALIZATION_H

#include "matrix.h"

typedef enum {
    INIT_XAVIER, //normal
    INIT_XAVIER_UNIFORM, //uniform
    INIT_HE, //normal
    INIT_HE_UNIFORM //uniform
} InitMethod;

void set_seed(unsigned int seed);
void mat_init_weights(Matrix *m, InitMethod method, int n_in, int n_out);

#endif
