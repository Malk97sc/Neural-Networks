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

int main()
{
    runtime_init(4); // number of threads

    printf("Running performance tests...\n\n");

    /* Mat-Vec */
    test_matvec_runtime(2, 2);
    test_matvec_runtime(4000, 4000); // parallel region

    /* Mat-Mat */
    test_matmul_runtime(2, 2, 2);
    test_matmul_runtime(400, 400, 400); // parallel region

    printf("\nAll performance tests completed.\n");
    runtime_destroy();
    return 0;
}

static int float_eq(float a, float b)
{
    return fabsf(a - b) < EPS;
}

static double now()
{
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

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
            MAT_AT(&A, i, j) = 1.0f;
    }

    /* --- Sequential --- */
    double t0 = now();

    matvec(&A, x, y_seq);  // puede usar runtime → lo forzamos abajo

    double t1 = now();

    /* --- Parallel (forzado) --- */
    const RuntimeConfig *cfg = runtime_get();

    double t2 = now();

    matvec_parallel(A.data ? &A : NULL, x, y_par, cfg->n_threads);

    double t3 = now();

    /* --- Correctness --- */
    for (int i = 0; i < rows; i++)
    {
        assert(float_eq(y_seq[i], (float)cols));
        assert(float_eq(y_par[i], (float)cols));
    }

    double t_seq = t1 - t0;
    double t_par = t3 - t2;
    double speedup = t_seq / t_par;

    printf("  seq: %.6f s\n", t_seq);
    printf("  par: %.6f s\n", t_par);
    printf("  speedup: %.2fx\n\n", speedup);

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

    /* --- Sequential --- */
    double t0 = now();

    matmul(&A, &B, &C_seq);

    double t1 = now();

    /* --- Parallel --- */
    const RuntimeConfig *cfg = runtime_get();

    double t2 = now();

    matmul_parallel(&A, &B, &C_par, cfg->n_threads);

    double t3 = now();

    /* --- Correctness --- */
    for (int i = 0; i < m * p; i++)
    {
        assert(float_eq(C_seq.data[i], (float)n));
        assert(float_eq(C_par.data[i], (float)n));
    }

    double t_seq = t1 - t0;
    double t_par = t3 - t2;
    double speedup = t_seq / t_par;

    printf("  seq: %.6f s\n", t_seq);
    printf("  par: %.6f s\n", t_par);
    printf("  speedup: %.2fx\n\n", speedup);

    mat_free(&A);
    mat_free(&B);
    mat_free(&C_seq);
    mat_free(&C_par);
}