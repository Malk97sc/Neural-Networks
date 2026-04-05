#ifndef RUNTIME_H
#define RUNTIME_H

#include "thread_pool.h"

#define MIN_THREADS 2

typedef struct {
    int n_threads;
    int matvec_threshold;
    int matmul_threshold;
    int elementwise_add_threshold;
    int elementwise_threshold;
    ThreadPool *pool;
} RuntimeConfig;

void runtime_init(int n_threads);
void runtime_destroy(void);
const RuntimeConfig* runtime_get(void);

int should_parallelize_matvec(int rows, int cols);
int should_parallelize_matmul(int m, int n, int p);
int should_parallelize_elementwise_add(int rows, int cols);
int should_parallelize_elementwise(int size);

#endif