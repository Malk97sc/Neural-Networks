#include <pthread.h>
#include <assert.h>

#include "linalg.h"
#include "parallel.h"

typedef struct {
    const Matrix *A;
    const Matrix *BT; //I'm gonna use the B transposed to reduce cache misses
    Matrix *C;

    int start_row;
    int end_row;
} MatMulTask;

static void *worker(void *arg){
    MatMulTask *task = (MatMulTask *)arg;
    const Matrix *A, *BT;
    Matrix *C;
    const float *rowA, *rowB;

    A = task->A;
    BT = task->BT;
    C = task->C;

    for(int i = task->start_row; i < task->end_row; i++){
        rowA = &A->data[i * A->stride];
        for(int j=0; j < BT->rows; j++){
            rowB = &BT->data[j * BT->stride];
            C->data[i * C->stride + j] = vec_dot(rowA, rowB, A->cols);
        }
    }

    return NULL;
}

void matmul_parallel_impl(const Matrix *A, const Matrix *B, Matrix *C, int n_threads){
    assert(A && B && C);
    assert((A->cols == B->rows) && (C->rows == A->rows) && (C->cols == B->cols));

    Matrix BT;
    int rows, chunk;
    pthread_t threads[n_threads];
    MatMulTask tasks[n_threads];

    rows = A->rows;
    chunk = rows / n_threads;

    BT = mat_alloc(B->cols, B->rows);
    mat_transpose(B, &BT);

    for(int t=0; t < n_threads; t++){
        tasks[t].A = A;
        tasks[t].BT = &BT;
        tasks[t].C = C;
        tasks[t].start_row = chunk * t;
        tasks[t].end_row = (t == n_threads - 1) ? rows : chunk * (t+1);

        pthread_create(&threads[t], NULL, worker, &tasks[t]);
    }

    for(int t = 0; t < n_threads; t++) pthread_join(threads[t], NULL);

    mat_free(&BT);
}