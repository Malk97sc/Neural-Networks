#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "linalg.h"
#include "parallel.h"
#include "runtime.h"

float vec_dot(const float *__restrict a, const float * __restrict b, int n){
    assert(a != NULL && b != NULL);
    assert(n >= 0);

    float sum = 0.0f;

    for(int i=0; i < n; i++){
        sum += a[i] * b[i];
    }

    return sum;
}

void vec_add(const float * __restrict a, const float * __restrict b, float * __restrict out, int n){
    assert(a && b && out);
    assert(n >= 0);

    for(int i=0; i < n; i++){
        out[i] = a[i] + b[i];
    }
}

void vec_scale(float *x, float alpha, int n){
    assert(x != NULL);
    assert(n >= 0);

    for(int i=0; i < n; i++) x[i] *= alpha;
}

/*2. (matrix-vector) */
void matvec(const Matrix *A, const float *x, float *y){
    assert(A && x && y);
    assert(A->cols >= 0 && A->rows >= 0);

    if(should_parallelize_matvec(A->rows, A->cols)){
        printf("[matvec] parallel path\n");
        const RuntimeConfig *cfg = runtime_get();
        matvec_parallel_impl(A, x, y, cfg->n_threads);
        return;
    }

    const float *row = NULL;

    for(int i=0; i < A->rows; i++){
        row = &A->data[i * A->stride]; //find the row
        y[i] = vec_dot(row, x, A->cols);
    }
}

void matvec_parallel(const Matrix *A, const float *x, float *y, int n_threads){
    matvec_parallel_impl(A, x, y, n_threads);
}

/*3. (matrix-matrix) */
void matmul(const Matrix *A, const Matrix *B, Matrix *C){
    assert(A && B && C);
    assert((A->cols == B->rows) && (C->rows == A->rows) && (C->cols == B->cols));

    if(should_parallelize_matmul(A->rows, A->cols, B->cols)){
        printf("[matmul] parallel path\n");
        const RuntimeConfig *cfg = runtime_get();
        matmul_parallel_impl(A, B, C, cfg->n_threads);
        return;
    }

    int rowsA, colsA, colsB;
    float sum = 0.0f;

    rowsA = A->rows;
    colsA = A->cols;
    colsB = B->cols;

    for (int i=0; i < rowsA; i++){
        for (int j=0; j < colsB; j++){
            sum = 0.0f;
            for (int k=0; k < colsA; k++){
                sum += A->data[i * A->stride + k] * B->data[k * B->stride + j];
                //sum += MAT_AT(A, i, k) * MAT_AT(B, k, j);
            }
            C->data[i * C->stride + j] = sum;
        }
    }
}

void matmul_parallel(const Matrix *A, const Matrix *B, Matrix *C, int n_threads){
    matmul_parallel_impl(A, B, C, n_threads);
}

/* transformations */
void mat_transpose(const Matrix *A, Matrix *AT){
    assert(A && AT);
    assert((AT->rows == A->cols) && (AT->cols == A->rows));

    for(int i=0; i < A->rows; i++){
        for(int j=0; j < A->cols; j++){
            AT->data[j * AT->stride + i] = A->data[i * A->stride + j];
        }
    }
}

/* Bias ops */
void mat_add_rowwise(Matrix *A, const float *b){
    assert(A && b);

    if(should_parallelize_elementwise_add(A->rows, A->cols)){
        const RuntimeConfig *cfg = runtime_get();
        mat_add_rowwise_parallel_impl(A, b, cfg->n_threads);
        return;
    }

    float *row = NULL;

    for(int i=0; i < A->rows; i++){
        row = &A->data[i * A->stride];
        for(int j=0; j < A->cols; j++){
            row[j] += b[j];
        }
    }
}

void mat_add_rowwise_parallel(Matrix *A, const float *b, int n_threads){
    mat_add_rowwise_parallel_impl(A, b, n_threads);
}

/* element-wise ops */
void mat_apply(Matrix *A, float (*func)(float)){
    assert(A && func);

    if(should_parallelize_elementwise(A->rows * A->cols)){
        const RuntimeConfig *cfg = runtime_get();
        mat_apply_parallel_impl(A, func, cfg->n_threads);
        return;
    }

    float *row = NULL;

    for(int i=0; i < A->rows; i++){
        row = &A->data[i * A->stride];
        for(int j=0; j < A->cols; j++){
            row[j] = func(row[j]);
        }
    }
}

void mat_apply_parallel(Matrix *A, float (*func)(float), int n_threads){
    mat_apply_parallel_impl(A, func, n_threads);
}