#ifndef LOSS_H
#define LOSS_H

#include "matrix.h"

float mse(const Matrix *y_true, const Matrix *y_pred);
void mse_grad(const Matrix *y_true, const Matrix *y_pred, Matrix *grad);

#endif
