#include <stddef.h>
#include "runtime.h"
#include "thread_pool.h"

static RuntimeConfig config;

void runtime_init(int n_threads){
    config.n_threads = n_threads;

    //main thresholds
    config.matvec_threshold = 200000;  // rows * cols
    config.matmul_threshold = 5000000; // Arows * Acols * Bcols
    config.elementwise_add_threshold = 200000;  // rows * cols
    config.elementwise_threshold = 100000;  // rows * cols

    config.pool = thread_pool_create(n_threads);
}

void runtime_destroy(void){
    if(config.pool){
        thread_pool_destroy(config.pool);
        config.pool = NULL;
    }
}

const RuntimeConfig* runtime_get(void){
    return &config;
}

int should_parallelize_matvec(int rows, int cols){
    return (rows * cols) >= config.matvec_threshold;
}

int should_parallelize_matmul(int m, int n, int p){
    return (m * n * p) >= config.matmul_threshold;
}

int should_parallelize_elementwise_add(int rows, int cols){
    return (rows * cols) >= config.elementwise_add_threshold;
}

int should_parallelize_elementwise(int size){
    return size >= config.elementwise_threshold;
}