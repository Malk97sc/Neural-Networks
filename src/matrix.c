#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "matrix.h"

Matrix mat_alloc(int rows, int cols){
    assert(rows >= 0 && cols >= 0);
    
    Matrix mtx;
    mtx.rows = rows;
    mtx.stride = mtx.cols = cols;
    if(rows == 0 || cols == 0){
        mtx.data = NULL;
        return mtx;
    }

    mtx.data = (float*) malloc(rows * cols * sizeof(float));
    if(!mtx.data){
        perror("Fail malloc\n");
        exit(-1);
    }

    return mtx;
}

void mat_free(Matrix *mtx){
    assert(mtx != NULL);

    if(mtx->data != NULL){
        free(mtx->data);
        mtx->data = NULL;
    }

    mtx->rows = mtx->cols = mtx->stride = 0;
}

void mat_fill(Matrix *mtx, float value){
    assert(mtx != NULL);

    int total;

    total = mtx->rows * mtx->stride;

    for(int i=0; i < total; i++){
        mtx->data[i] = value;
    }
}

void mat_copy(Matrix *dst, const Matrix *src){
    assert(dst != NULL && src != NULL);
    assert((dst->rows == src->rows) && (dst->cols == src->cols));

    const float *src_row;
    float *dest_row;

    for(int i=0; i < src->rows; i++){
        src_row = &src->data[i * src->stride];
        dest_row = &dst->data[i * dst->stride];

        for (int j = 0; j < src->cols; j++){
            dest_row[j] = src_row[j];
        }
    }
}

void mat_print(const Matrix *mtx){
    assert(mtx != NULL);

    for(int i=0; i < mtx->rows; i++){
        for(int j=0; j < mtx->cols; j++){
            printf("%8.4f ", MAT_AT(mtx, i, j));
        }
        printf("\n");
    }
}


