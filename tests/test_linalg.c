#include <stdio.h>
#include <assert.h>
#include <math.h>

#include "matrix.h"
#include "linalg.h"
#include "runtime.h"
#include "activations.h"

#define EPS 1e-5f

int float_eq(float a, float b);
void test_vec_ops();
void test_matvec();
void test_matmul();
void test_transpose();
void test_bias();
void test_apply();

int main(){
    runtime_init(4);

    printf("Running tests...\n");

    test_vec_ops();
    test_matvec();
    test_matmul();
    test_transpose();
    test_bias();
    test_apply();

    printf("All tests passed.\n");

    runtime_destroy();
    return 0;
}

int float_eq(float a, float b){
    return fabsf(a - b) < EPS;
}

void test_vec_ops(){
    float a[3] = {1, 2, 3};
    float b[3] = {4, 5, 6};
    float out[3];

    float d = vec_dot(a, b, 3);
    assert(float_eq(d, 32.0f));

    vec_add(a, b, out, 3);
    assert(float_eq(out[0], 5));
    assert(float_eq(out[1], 7));
    assert(float_eq(out[2], 9));

    vec_scale(a, 2.0f, 3);
    assert(float_eq(a[0], 2));
    assert(float_eq(a[1], 4));
    assert(float_eq(a[2], 6));
}

void test_matvec(){
    Matrix A = mat_alloc(2, 2);

    MAT_AT(&A, 0, 0) = 1; MAT_AT(&A, 0, 1) = 0;
    MAT_AT(&A, 1, 0) = 0; MAT_AT(&A, 1, 1) = 1;

    float x[2] = {7, 9};
    float y[2] = {0};

    matvec(&A, x, y);

    assert(float_eq(y[0], 7));
    assert(float_eq(y[1], 9));

    mat_free(&A);
}

void test_matmul(){
    Matrix A = mat_alloc(2, 3);
    Matrix B = mat_alloc(3, 2);
    Matrix C = mat_alloc(2, 2);

    float a_vals[] = {1,2,3,4,5,6};
    float b_vals[] = {7,8,9,10,11,12};

    for (int i = 0; i < 6; i++) A.data[i] = a_vals[i];
    for (int i = 0; i < 6; i++) B.data[i] = b_vals[i];

    matmul(&A, &B, &C);

    assert(float_eq(MAT_AT(&C, 0, 0), 58));
    assert(float_eq(MAT_AT(&C, 0, 1), 64));
    assert(float_eq(MAT_AT(&C, 1, 0), 139));
    assert(float_eq(MAT_AT(&C, 1, 1), 154));

    mat_free(&A);
    mat_free(&B);
    mat_free(&C);
}

void test_transpose(){
    Matrix A = mat_alloc(2, 3);
    Matrix AT = mat_alloc(3, 2);

    for (int i = 0; i < 6; i++)
        A.data[i] = (float)(i + 1);

    mat_transpose(&A, &AT);

    assert(float_eq(MAT_AT(&AT, 0, 0), 1));
    assert(float_eq(MAT_AT(&AT, 1, 0), 2));
    assert(float_eq(MAT_AT(&AT, 2, 0), 3));
    assert(float_eq(MAT_AT(&AT, 0, 1), 4));
    assert(float_eq(MAT_AT(&AT, 1, 1), 5));
    assert(float_eq(MAT_AT(&AT, 2, 1), 6));

    mat_free(&A);
    mat_free(&AT);
}

void test_bias(){
    Matrix A = mat_alloc(2, 3);
    mat_fill(&A, 1.0f);

    float b[3] = {1, 2, 3};

    mat_add_rowwise(&A, b);

    assert(float_eq(MAT_AT(&A, 0, 0), 2));
    assert(float_eq(MAT_AT(&A, 0, 1), 3));
    assert(float_eq(MAT_AT(&A, 0, 2), 4));

    assert(float_eq(MAT_AT(&A, 1, 0), 2));
    assert(float_eq(MAT_AT(&A, 1, 1), 3));
    assert(float_eq(MAT_AT(&A, 1, 2), 4));

    mat_free(&A);
}

void test_apply(){
    Matrix A = mat_alloc(2, 2);

    MAT_AT(&A, 0, 0) = -1;
    MAT_AT(&A, 0, 1) = 2;
    MAT_AT(&A, 1, 0) = -3;
    MAT_AT(&A, 1, 1) = 4;

    mat_apply(&A, relu);

    assert(float_eq(MAT_AT(&A, 0, 0), 0));
    assert(float_eq(MAT_AT(&A, 0, 1), 2));
    assert(float_eq(MAT_AT(&A, 1, 0), 0));
    assert(float_eq(MAT_AT(&A, 1, 1), 4));

    mat_free(&A);
}