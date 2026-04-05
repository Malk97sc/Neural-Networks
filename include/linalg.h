#ifndef LINALG_H
#define LINALG_H

#include "matrix.h"

/*1 (vector) */
float vec_dot(const float *__restrict a, const float * __restrict b, int n);
void vec_add(const float * __restrict a, const float * __restrict b, float * __restrict out, int n);
void vec_scale(float *x, float alpha, int n);

/*2. (matrix-vector) */
void matvec(const Matrix *A, const float *x, float *y);
void matvec_parallel(const Matrix *A, const float *x, float *y, int n_threads);

/*3. (matrix-matrix) */
void matmul(const Matrix *A, const Matrix *B, Matrix *C);
void matmul_parallel(const Matrix *A, const Matrix *B, Matrix *C, int n_threads);

/* transformations */
void mat_transpose(const Matrix *A, Matrix *AT);

/* Bias ops */
void mat_add_rowwise(Matrix *A, const float *b); 
void mat_add_rowwise_parallel(Matrix *A, const float *b, int n_threads);

/* element-wise ops */
void mat_apply(Matrix *A, float (*func)(float));
void mat_apply_parallel(Matrix *A, float (*func)(float), int n_threads);

void mat_apply_binary(Matrix *A, const Matrix *B, float (*func)(float, float));
void mat_apply_binary_parallel(Matrix *A, const Matrix *B, float (*func)(float, float), int n_threads);

#endif