#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include "matrix.h"

float relu(float x);
float dx_relu(float x);

float sigmoid(float x);
float dx_sigmoid(float x);

float nn_tanh(float x);
float dx_nn_tanh(float x);

/*matrix applications*/
void mat_relu(Matrix *m);
void mat_dx_relu(Matrix *grad, const Matrix *inp_z);

void mat_sigmoid(Matrix *m);
void mat_dx_sigmoid(Matrix *grad, const Matrix *out);

void mat_nn_tanh(Matrix *m);
void mat_dx_nn_tanh(Matrix *grad, const Matrix *out);

#endif
