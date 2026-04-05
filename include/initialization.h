#ifndef INITIALIZATION_H
#define INITIALIZATION_H

#include "matrix.h"

typedef enum {
    INIT_XAVIER,
    INIT_HE
} InitMethod;

void set_seed(unsigned int seed);
void mat_init_weights(Matrix *m, InitMethod method, int n_in, int n_out);

#endif
