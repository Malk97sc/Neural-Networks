#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "slp/slp.h"
#include "linalg.h"

static float step_activation(float x){
    return (x >= 0.0f) ? 1.0f : 0.0f;
}

SLP* slp_create(int in_dim, InitMethod init){
    int n_out = 1;

    SLP *net = (SLP*)malloc(sizeof(SLP));
    net->in_dim = in_dim;
    
    net->weights = mat_alloc(in_dim, n_out);
    mat_init_weights(&(net->weights), init, in_dim, n_out);
    
    net->bias = 0.0f;
    net->y_pred = 0.0f;
    
    return net;
}

void slp_free(SLP *net){
    mat_free(&(net->weights));
    free(net);
}

float slp_forward(SLP *net, const Matrix *input){
    assert(input->cols == net->in_dim);
    float z_val;
    Matrix z_mat;
    
    z_mat = mat_alloc(1, 1); // Z = input * W. [1, in_dim] * [in_dim, 1] = [1, 1]
    matmul(input, &(net->weights), &z_mat);
    
    z_val = z_mat.data[0] + net->bias;
    mat_free(&z_mat);
    
    // y_pred = step(z)
    net->y_pred = step_activation(z_val);
    return net->y_pred;
}

float slp_train_step(SLP *net, const Matrix *input, float target, float lr){
    float y_pred, error;
    
    y_pred = slp_forward(net, input); // (Forward)
    
    error = target - y_pred;
    
    // W += lr * error * x
    // b += lr * error
    if(error != 0.0f){
        for(int i=0; i < net->in_dim; i++){
            net->weights.data[i] += lr * error * input->data[i];
        }
        
        net->bias += lr * error;
    }
    
    return error;
}
