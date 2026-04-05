#include <assert.h>

#include "parallel.h"
#include "linalg.h"
#include "runtime.h"
#include "thread_pool.h"

typedef struct {
    Matrix *A;
    const Matrix *B;
    float (*func)(float, float);
    int start_row;
    int end_row;
} ApplyBinaryTask;

static void worker(void *arg){
    ApplyBinaryTask *task = (ApplyBinaryTask *)arg;
    Matrix *A;
    const Matrix *B;
    float (*func)(float, float);
    float *rowA;
    const float *rowB;

    A = task->A;
    B = task->B;
    func = task->func;

    for(int i = task->start_row; i < task->end_row; i++){
        rowA = &A->data[i * A->stride];
        rowB = &B->data[i * B->stride];
        for(int j = 0; j < A->cols; j++){
            rowA[j] = func(rowA[j], rowB[j]);
        }            
    }
}

void mat_apply_binary_parallel_impl(Matrix *A, const Matrix *B, float (*func)(float, float), int n_threads){
    assert(A && B && func);
    assert(A->rows == B->rows && A->cols == B->cols);
    assert(n_threads > 0);

    if(n_threads == 1 || A->rows < n_threads){
        mat_apply_binary(A, B, func);
        return;
    }

    int rows, chunk;
    ApplyBinaryTask tasks[n_threads];
    void *args[n_threads];

    rows = A->rows;
    chunk = rows / n_threads;

    for(int t = 0; t < n_threads; t++){
        tasks[t].A = A;
        tasks[t].B = B;
        tasks[t].func = func;
        tasks[t].start_row = chunk * t;
        tasks[t].end_row = (t == n_threads - 1) ? rows : chunk * (t + 1);
        args[t] = &tasks[t];
    }

    const RuntimeConfig *cfg = runtime_get();
    thread_pool_submit(cfg->pool, worker, args, n_threads);
    thread_pool_wait(cfg->pool);
}
