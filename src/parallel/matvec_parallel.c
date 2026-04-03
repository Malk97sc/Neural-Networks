#include <assert.h>

#include "linalg.h"
#include "parallel.h"
#include "runtime.h"
#include "thread_pool.h"

typedef struct {
    const Matrix *A;
    const float *x;
    float *y;
    int start_row;
    int end_row;
} MatVecTask;

static void worker(void *arg){
    MatVecTask *task = (MatVecTask *)arg;
    const Matrix *A;
    const float *x;
    float *y;
    const float *row;

    A = task->A;
    x = task->x;
    y = task->y;

    for(int i = task->start_row; i < task->end_row; i++){
        row = &A->data[i * A->stride];
        y[i] = vec_dot(row, x, A->cols);
    }
}

void matvec_parallel_impl(const Matrix *A, const float *x, float *y, int n_threads){
    assert(A && x && y);
    assert(n_threads > 0);

    if(n_threads == 1 || A->rows < n_threads){
        matvec(A, x, y);
        return;
    }

    int rows, chunk;
    MatVecTask tasks[n_threads];
    void *args[n_threads];

    rows = A->rows;
    chunk = rows / n_threads;

    for(int t = 0; t < n_threads; t++){
        tasks[t].A = A;
        tasks[t].x = x;
        tasks[t].y = y;
        tasks[t].start_row = chunk * t;
        tasks[t].end_row = (t == n_threads - 1) ? rows : chunk * (t + 1);
        args[t] = &tasks[t];
    }

    const RuntimeConfig *cfg = runtime_get();
    thread_pool_submit(cfg->pool, worker, args, n_threads);
    thread_pool_wait(cfg->pool);
}