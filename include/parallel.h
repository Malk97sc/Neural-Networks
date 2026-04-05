#ifndef PARALLEL_H
#define PARALLEL_H

#include "matrix.h"

void matvec_parallel_impl(const Matrix *A, const float *x, float *y, int n_threads);
void matmul_parallel_impl(const Matrix *A, const Matrix *B, Matrix *C, int n_threads);

void mat_add_rowwise_parallel_impl(Matrix *A, const float *b, int n_threads);
void mat_apply_parallel_impl(Matrix *A, float (*func)(float), int n_threads);

void mat_apply_binary_parallel_impl(Matrix *A, const Matrix *B, float (*func)(float, float), int n_threads);

#endif