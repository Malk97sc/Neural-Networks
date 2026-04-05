#ifndef SLP_H
#define SLP_H

#include "matrix.h"
#include "initialization.h"

typedef struct {
    int in_dim;
    
    Matrix weights;
    float bias;
    
    float y_pred;
} SLP;

SLP* slp_create(int in_dim, InitMethod init);
void slp_free(SLP *net);

float slp_forward(SLP *net, const Matrix *input);
float slp_train_step(SLP *net, const Matrix *input, float target, float lr);

#endif
