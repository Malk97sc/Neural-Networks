#include <pthread.h>
#include <assert.h>

#include "parallel.h"
#include "linalg.h"

typedef struct{
    Matrix *A;
    const float *b;
    int start_row;
    int end_row;
} BiasTask;

static void *worker(void *arg){
    BiasTask *task = (BiasTask *)arg;
    Matrix *A;
    const float *b;
    float *row;

    A = task->A;
    b = task->b;

    for(int i = task->start_row; i < task->end_row; i++){
        row = &A->data[i * A->stride];
        for(int j=0; j < A->cols; j++){
            row[j] += b[j];
        }
    }

    return NULL;
}

void mat_add_rowwise_parallel_impl(Matrix *A, const float *b, int n_threads){
    assert(A && b);
    assert(n_threads > 0);

    if(n_threads == 1 || A->rows < n_threads){
        mat_add_rowwise(A, b);
        return;
    }

    int rows, chunk;
    pthread_t threads[n_threads];
    BiasTask tasks[n_threads];

    rows = A->rows;
    chunk = rows / n_threads;

    for(int t = 0; t < n_threads; t++){
        tasks[t].A = A;
        tasks[t].b = b;
        tasks[t].start_row = chunk * t;
        tasks[t].end_row = (t == n_threads - 1) ? rows : chunk * (t+1);

        pthread_create(&threads[t], NULL, worker, &tasks[t]);
    }

    for(int t = 0; t < n_threads; t++) pthread_join(threads[t], NULL);
}