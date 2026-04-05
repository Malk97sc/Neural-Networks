#include <stdio.h>
#include <stdlib.h>
#include "slp/slp.h"
#include "runtime.h"
#include <math.h>

void show_data(int in_dim, int size, float inputs[][in_dim], float targets[]){
    printf("\nData:\n");
    printf("   x    | y\n");
    printf("------------\n");
    for(int i=0; i<size; i++){
        printf("[ ");
        for(int j=0; j<in_dim; j++){
            printf("%d ", (int)inputs[i][j]);
        }
        printf("] | %d\n", (int)targets[i]);
    }
    printf("\n");
}

void test_perceptron_and_gate() {
    printf("--- Rosenblatt Perceptron: AND Training ---\n");
    
    //AND
    int size = 4;
    float inputs[4][2] = {{0.0f, 0.0f}, 
                          {0.0f, 1.0f}, 
                          {1.0f, 0.0f}, 
                          {1.0f, 1.0f}};
    float targets[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    
    float lr = 0.1f, pred, total_epoch_error;
    int max_epochs = 20;

    show_data(2, size, inputs, targets);

    SLP *net = slp_create(2, INIT_XAVIER);
    Matrix x = mat_alloc(1, 2);

    printf("Starting Training\n");
    for(int epoch = 0; epoch < max_epochs; epoch++) {
        total_epoch_error = 0;
        for(int i = 0; i < size; i++) {
            x.data[0] = inputs[i][0];
            x.data[1] = inputs[i][1]; 

            pred = slp_train_step(net, &x, targets[i], lr);
            total_epoch_error += fabsf(pred); 
        }
        
        if(epoch % 5 == 0) printf("Epoch %d, Total Epoch Errors: %.6f\n", epoch, total_epoch_error);
    }

    
    printf("\nPredictions for AND Gate:\n");
    for(int i = 0; i < size; i++) {
        x.data[0] = inputs[i][0]; x.data[1] = inputs[i][1];
        pred = slp_forward(net, &x);
        printf("[%g, %g] -> Expect %g, Got %.0f\n", inputs[i][0], inputs[i][1], targets[i], pred);
    }

    mat_free(&x);
    slp_free(net);
}

int main() {
    runtime_init(4);
    test_perceptron_and_gate();
    runtime_destroy();
    return 0;
}
