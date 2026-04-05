#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <stdlib.h>

#include "activations.h"
#include "loss.h"
#include "initialization.h"
#include "matrix.h"
#include "runtime.h"

void test_activations();
void test_loss_grad();
void test_initialization();

int main(){
    runtime_init(4);

    test_activations();
    test_loss_grad();
    test_initialization();

    runtime_destroy();
    return 0;
}

void test_activations(){
    printf("--- Testing Activations ---\n");
    float x = -2.0, y = 2.0;

    printf("ReLU(%f) = %f, ReLU(%f) = %f\n", x, relu(x), y, relu(y));
    printf("Sigmoid(0) = %f (expect 0.5)\n", sigmoid(0.0));
    printf("Tanh(0) = %f\n", nn_tanh(0.0));

    // Derivatives
    printf("ReLU'(%f) = %f, ReLU'(%f) = %f\n", x, dx_relu(x), y, dx_relu(y));
    printf("Sigmoid'(0) = %f\n", dx_sigmoid(0.0));
    printf("Tanh'(0) = %f\n", dx_nn_tanh(0.0));
}

void test_loss_grad(){
    printf("\n--- Testing MSE Gradient ---\n");
    Matrix y_true = mat_alloc(1, 2);
    Matrix y_pred = mat_alloc(1, 2);
    Matrix grad = mat_alloc(1, 2);

    y_true.data[0] = 1.0; y_true.data[1] = 0.0;
    y_pred.data[0] = 0.8; y_pred.data[1] = 0.2;

    float loss = mse(&y_true, &y_pred);
    mse_grad(&y_true, &y_pred, &grad);

    printf("Loss (MSE): %f\n", loss);
    printf("Grad: [%f, %f]\n", grad.data[0], grad.data[1]);
    
    float epsilon = 1e-4;
    assert(fabs(loss - 0.04f) < epsilon);
    assert(fabs(grad.data[0] - (-0.2f)) < epsilon);
    assert(fabs(grad.data[1] - 0.2f) < epsilon);

    mat_free(&y_true);
    mat_free(&y_pred);
    mat_free(&grad);
}

void test_initialization(){
    printf("\n--- Testing Initialization ---\n");
    Matrix w = mat_alloc(100, 100);
    set_seed(42);
    mat_init_weights(&w, INIT_XAVIER, 100, 100);
    
    float sum = 0, sq_sum = 0;
    int n = w.rows * w.cols;
    for(int i=0; i<n; i++) {
        sum += w.data[i];
        sq_sum += w.data[i] * w.data[i];
    }
    float mean = sum / n;
    float var = (sq_sum / n) - (mean * mean);
    printf("Xavier (100,100) -> Mean: %f, Var: %f (Expected ~0.01)\n", mean, var);
    
    mat_free(&w);
}