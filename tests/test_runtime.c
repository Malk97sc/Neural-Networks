#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include "matrix.h"
#include "linalg.h"
#include "runtime.h"

#define EPS 1e-5f

static int float_eq(float a, float b);
void test_matvec_runtime(int rows, int cols);
void test_matmul_runtime(int m, int n, int p);

int main(){
    runtime_init(4); //number of threads

    printf("Running runtime validation with different sizes...\n");

    //Mat-Vec
    test_matvec_runtime(2, 2);
    test_matvec_runtime(15000, 15000); //parallel

    //Mat-Mat
    test_matmul_runtime(2, 2, 2); 
    test_matmul_runtime(2000, 2000, 2000); //parallel

    printf("All runtime tests passed.\n");
    runtime_destroy();
    return 0;
}


static int float_eq(float a, float b){
    return fabsf(a - b) < EPS;
}

void test_matvec_runtime(int rows, int cols){
    Matrix A = mat_alloc(rows, cols);
    float *x = malloc(sizeof(float) * cols);
    float *y = malloc(sizeof(float) * rows);

    for (int j = 0; j < cols; j++) x[j] = 1.0f;

    for (int i = 0; i < rows; i++){
        y[i] = 0.0f;
        for (int j = 0; j < cols; j++){
            MAT_AT(&A, i, j) = 1.0f;
        }
    }

    matvec(&A, x, y);

    for (int i = 0; i < rows; i++){
        assert(float_eq(y[i], (float)cols));
    }

    free(x);
    free(y);
    mat_free(&A);

    printf("matvec %dx%d OK\n", rows, cols);
}

void test_matmul_runtime(int m, int n, int p){
    Matrix A = mat_alloc(m, n);
    Matrix B = mat_alloc(n, p);
    Matrix C = mat_alloc(m, p);

    mat_fill(&A, 1.0f);
    mat_fill(&B, 1.0f);

    matmul(&A, &B, &C);

    for (int i = 0; i < m * p; i++){
        assert(float_eq(C.data[i], (float)n));
    }

    mat_free(&A);
    mat_free(&B);
    mat_free(&C);

    printf("matmul %dx%dx%d OK\n", m, n, p);
}
