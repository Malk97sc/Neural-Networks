#include <math.h>
#include <assert.h>
#include "activations.h"
#include "linalg.h"

float relu(float x){
    return x > 0 ? x : 0.0f;
}

float dx_relu(float x){
    return x > 0 ? 1.0f : 0.0f;
}

float sigmoid(float x){
    return 1.0f / (1.0f + expf(-x));
}

float dx_sigmoid(float x){
    float s = sigmoid(x);
    return s * (1.0f - s);
}

float nn_tanh(float x){
    return tanhf(x);
}

float dx_nn_tanh(float x){
    float t = tanhf(x);
    return 1.0f - t * t;
}

void mat_relu(Matrix *m){
    mat_apply(m, relu);
}

void mat_sigmoid(Matrix *m){
    mat_apply(m, sigmoid);
}

void mat_nn_tanh(Matrix *m){
    mat_apply(m, nn_tanh);
}

//all static functions are for mat_apply_binary (used by the threads)
static float dx_relu_binary(float grad_val, float inp_z_val){
    return inp_z_val <= 0 ? 0.0f : grad_val;
}

static float dx_sigmoid_binary(float grad_val, float out_val){
    return grad_val * out_val * (1.0f - out_val);
}

static float dx_nn_tanh_binary(float grad_val, float out_val){
    return grad_val * (1.0f - out_val * out_val);
}

void mat_dx_sigmoid(Matrix *grad, const Matrix *out){
    mat_apply_binary(grad, out, dx_sigmoid_binary);
}

void mat_dx_relu(Matrix *grad, const Matrix *inp_z){
    mat_apply_binary(grad, inp_z, dx_relu_binary);
}

void mat_dx_nn_tanh(Matrix *grad, const Matrix *out){
    mat_apply_binary(grad, out, dx_nn_tanh_binary);
}




