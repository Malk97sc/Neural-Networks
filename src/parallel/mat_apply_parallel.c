#include <pthread.h>
#include <assert.h>

#include "parallel.h"
#include "linalg.h"

typedef struct{
    Matrix *A;
    float (*func)(float);
    int start_row;
    int end_row;
} ApplyTask;

static void *worker(void *arg){
    ApplyTask *task = (ApplyTask *)arg;
    Matrix *A;
    float (*func)(float);
    float *row;

    A = task->A;
    func = task->func;

    for(int i = task->start_row; i < task->end_row; i++){
        row = &A->data[i * A->stride];
        for(int j=0; j < A->cols; j++){
            row[j] = func(row[j]);
        }
    }

    return NULL;
}

void mat_apply_parallel_impl(Matrix *A, float (*func)(float), int n_threads){
    assert(A && func);
    assert(n_threads > 0);

    if(n_threads == 1 || A->rows < n_threads){
        mat_apply(A, func);
        return;
    }

    int rows, chunk;
    pthread_t threads[n_threads];
    ApplyTask tasks[n_threads];

    rows = A->rows;
    chunk = rows / n_threads;

    for(int t=0; t < n_threads; t++){
        tasks[t].A = A;
        tasks[t].func = func;
        tasks[t].start_row = chunk * t;
        tasks[t].end_row = (t == n_threads - 1) ? rows : chunk * (t+1);

        pthread_create(&threads[t], NULL, worker, &tasks[t]);
    }

    for(int t = 0; t < n_threads; t++) pthread_join(threads[t], NULL);
}