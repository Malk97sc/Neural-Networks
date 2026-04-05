#include <assert.h>
#include "loss.h"
#include "linalg.h"

static float mse_n_val = 1.0f;

float mse(const Matrix *y_true, const Matrix *y_pred){
    assert(y_true->rows == y_pred->rows && y_true->cols == y_pred->cols);
    float total_err = 0.0f, diff;
    int size;
    
    size = y_true->rows * y_true->stride;

    for(int i=0; i < size; i++){
        diff = y_true->data[i] - y_pred->data[i];
        total_err += diff * diff;
    }
    return total_err / (y_true->rows * y_true->cols);
}

static float mse_grad_binary(float pred_val, float true_val){
    return (2.0f / mse_n_val) * (pred_val - true_val);
}

void mse_grad(const Matrix *y_true, const Matrix *y_pred, Matrix *grad){
    assert(y_true->rows == y_pred->rows && y_true->cols == y_pred->cols);
    assert(grad->rows == y_true->rows && grad->cols == y_true->cols);

    mat_copy(grad, y_pred);
    
    mse_n_val = (float)(y_true->rows * y_true->cols);
    mat_apply_binary(grad, y_true, mse_grad_binary);
}
