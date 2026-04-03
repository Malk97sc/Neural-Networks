#include <assert.h>

#include "linalg.h"
#include "parallel.h"
#include "runtime.h"
#include "thread_pool.h"

typedef struct {
    const Matrix *A;
    const Matrix *BT; //I'm gonna use the B transposed to reduce cache misses
    Matrix *C;
    int start_row;
    int end_row;
} MatMulTask;

static void worker(void *arg){
    MatMulTask *task = (MatMulTask *)arg;
    const Matrix *A, *BT;
    Matrix *C;
    const float *rowA, *rowB;

    A = task->A;
    BT = task->BT;
    C = task->C;

    for(int i = task->start_row; i < task->end_row; i++){
        rowA = &A->data[i * A->stride];
        for(int j = 0; j < BT->rows; j++){
            rowB = &BT->data[j * BT->stride];
            C->data[i * C->stride + j] = vec_dot(rowA, rowB, A->cols);
        }
    }
}

void matmul_parallel_impl(const Matrix *A, const Matrix *B, Matrix *C, int n_threads){
    assert(A && B && C);
    assert((A->cols == B->rows) && (C->rows == A->rows) && (C->cols == B->cols));

    int rows, chunk;
    Matrix BT;
    MatMulTask tasks[n_threads];
    void *args[n_threads];

    rows = A->rows;
    chunk = rows / n_threads;

    BT = mat_alloc(B->cols, B->rows);
    mat_transpose(B, &BT);

    for(int t = 0; t < n_threads; t++){
        tasks[t].A = A;
        tasks[t].BT = &BT;
        tasks[t].C = C;
        tasks[t].start_row = chunk * t;
        tasks[t].end_row = (t == n_threads - 1) ? rows : chunk * (t + 1);
        args[t] = &tasks[t];
    }

    const RuntimeConfig *cfg = runtime_get();
    thread_pool_submit(cfg->pool, worker, args, n_threads);
    thread_pool_wait(cfg->pool);

    mat_free(&BT);
}