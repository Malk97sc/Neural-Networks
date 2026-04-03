#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <time.h>

#include "matrix.h"
#include "linalg.h"
#include "runtime.h"

#define EPS 1e-5f

static int float_eq(float a, float b);
static double now();

void test_matvec_runtime(int rows, int cols);
void test_matmul_runtime(int m, int n, int p);

int main(){
    runtime_init(4); // number of threads

    printf("Running performance tests (Real Sequential vs Parallel Pool)...\n\n");

    /* Mat-Vec */
    test_matvec_runtime(2000, 2000);
    test_matvec_runtime(10000, 10000); 

    /* Mat-Mat */
    test_matmul_runtime(200, 200, 200);
    test_matmul_runtime(1000, 1000, 1000); 

    printf("\nAll performance tests completed.\n");
    runtime_destroy();
    return 0;
}

static int float_eq(float a, float b){
    return fabsf(a - b) < EPS;
}

static double now(){
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

void test_matvec_runtime(int rows, int cols){
    printf("matvec %dx%d\n", rows, cols);

    Matrix A = mat_alloc(rows, cols);
    float *x = malloc(sizeof(float) * cols);
    float *y_seq = malloc(sizeof(float) * rows);
    float *y_par = malloc(sizeof(float) * rows);

    for (int j = 0; j < cols; j++) x[j] = 1.0f;
    for (int i = 0; i < rows; i++){
        for (int j = 0; j < cols; j++){
            MAT_AT(&A, i, j) = 1.0f;
        }
    }

    //sequential
    double t0 = now();
    for(int i = 0; i < rows; i++){
        const float *row = &A.data[i * A.stride];
        y_seq[i] = vec_dot(row, x, A.cols);
    }
    double t1 = now();

    //parallel
    const RuntimeConfig *cfg = runtime_get();
    double t2 = now();
    matvec_parallel(&A, x, y_par, cfg->n_threads);
    double t3 = now();

    for (int i = 0; i < rows; i++){
        assert(float_eq(y_seq[i], (float)cols));
        assert(float_eq(y_par[i], (float)cols));
    }

    double t_seq = t1 - t0;
    double t_par = t3 - t2;
    double speedup = t_seq / t_par;

    printf("seq: %.6f s\n", t_seq);
    printf("par: %.6f s\n", t_par);
    printf("speedup: %.2fx\n\n", speedup);

    free(x);
    free(y_seq);
    free(y_par);
    mat_free(&A);
}

void test_matmul_runtime(int m, int n, int p){
    printf("matmul %dx%dx%d\n", m, n, p);

    Matrix A = mat_alloc(m, n);
    Matrix B = mat_alloc(n, p);
    Matrix C_seq = mat_alloc(m, p);
    Matrix C_par = mat_alloc(m, p);

    mat_fill(&A, 1.0f);
    mat_fill(&B, 1.0f);

    //sequential
    double t0 = now();
    for (int i = 0; i < m; i++){
        for (int j = 0; j < p; j++){
            float sum = 0.0f;
            for (int k = 0; k < n; k++){
                sum += A.data[i * A.stride + k] * B.data[k * B.stride + j];
            }
            C_seq.data[i * C_seq.stride + j] = sum;
        }
    }
    double t1 = now();

    //parallel
    const RuntimeConfig *cfg = runtime_get();
    double t2 = now();
    matmul_parallel(&A, &B, &C_par, cfg->n_threads);
    double t3 = now();

    for (int i = 0; i < m * p; i++){
        assert(float_eq(C_seq.data[i], (float)n));
        assert(float_eq(C_par.data[i], (float)n));
    }

    double t_seq = t1 - t0;
    double t_par = t3 - t2;
    double speedup = t_seq / t_par;

    printf("seq: %.6f s\n", t_seq);
    printf("par: %.6f s\n", t_par);
    printf("speedup: %.2fx\n\n", speedup);

    mat_free(&A);
    mat_free(&B);
    mat_free(&C_seq);
    mat_free(&C_par);
}
