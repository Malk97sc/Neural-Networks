#include <stdio.h>
#include <assert.h>
#include <math.h>

#include "matrix.h"
#include "linalg.h"
#include "parallel.h"
#include "runtime.h"

#define EPS 1e-5f

int float_eq(float a, float b);
float relu(float x);
void test_matvec_parallel();
void test_matmul_parallel();
void test_bias_parallel();
void test_apply_parallel();

int main(){
    runtime_init(4);

    printf("Running parallel tests...\n");

    test_matvec_parallel();
    test_matmul_parallel();
    test_bias_parallel();
    test_apply_parallel();

    printf("All parallel tests passed.\n");

    runtime_destroy();
    return 0;
}

int float_eq(float a, float b){
    return fabsf(a - b) < EPS;
}

float relu(float x){
    return x > 0 ? x : 0;
}

void test_matvec_parallel(){
    int size = 4;
    Matrix A = mat_alloc(size, size);

    for(int i=0; i < size; i++){
        for(int j=0; j < size; j++){
            MAT_AT(&A, i, j) = (i == j) ? 1.0f : 0.0f;
        }
    }

    float x[4] = {1, 2, 3, 4};
    float y_seq[4] = {0};
    float y_par[4] = {0};

    matvec(&A, x, y_seq);
    matvec_parallel(&A, x, y_par, 2);

    for(int i=0; i < size; i++){
        assert(float_eq(y_seq[i], y_par[i]));
    }

    mat_free(&A);
}

void test_matmul_parallel(){
    Matrix A = mat_alloc(2, 3);
    Matrix B = mat_alloc(3, 2);
    Matrix C1 = mat_alloc(2, 2);
    Matrix C2 = mat_alloc(2, 2);

    float a_vals[6] = {1,2,3,4,5,6};
    float b_vals[6] = {7,8,9,10,11,12};

    for(int i=0; i < 6; i++) A.data[i] = a_vals[i];
    for(int i=0; i < 6; i++) B.data[i] = b_vals[i];

    matmul(&A, &B, &C1);
    matmul_parallel(&A, &B, &C2, 2);

    for(int i=0; i < 4; i++){
        assert(float_eq(C1.data[i], C2.data[i]));
    }        

    mat_free(&A);
    mat_free(&B);
    mat_free(&C1);
    mat_free(&C2);
}

void test_bias_parallel(){
    Matrix A1 = mat_alloc(2, 3);
    Matrix A2 = mat_alloc(2, 3);

    mat_fill(&A1, 1.0f);
    mat_fill(&A2, 1.0f);

    float b[3] = {1, 2, 3};

    mat_add_rowwise(&A1, b);
    mat_add_rowwise_parallel(&A2, b, 2);

    for(int i=0; i < 6; i++){
        assert(float_eq(A1.data[i], A2.data[i]));
    }

    mat_free(&A1);
    mat_free(&A2);
}

void test_apply_parallel(){
    Matrix A1 = mat_alloc(2, 2);
    Matrix A2 = mat_alloc(2, 2);

    float vals[4] = {-1, 2, -3, 4};

    for (int i = 0; i < 4; i++){
        A1.data[i] = vals[i];
        A2.data[i] = vals[i];
    }

    mat_apply(&A1, relu);
    mat_apply_parallel(&A2, relu, 2);

    for (int i = 0; i < 4; i++){
        assert(float_eq(A1.data[i], A2.data[i]));
    }        

    mat_free(&A1);
    mat_free(&A2);
}