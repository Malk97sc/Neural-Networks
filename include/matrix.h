#ifndef MATRIX_H
#define MATRIX_H

#ifdef DEBUG
    #include <assert.h>
    #define MAT_AT(m, i, j) \
        (assert((i) < (m)->rows && (j) < (m)->cols), \
        (m)->data[(i) * (m)->stride + (j)])
#else
    #define MAT_AT(m, i, j) ((m)->data[(i) * (m)->stride + (j)])
#endif

typedef struct {
    float *data;
    int rows;
    int cols;
    int stride;  //(row-major)
} Matrix;

/* allocation */
Matrix mat_alloc(int rows, int cols);
void mat_free(Matrix *m);

/* utilities */
void mat_fill(Matrix *m, float value);
void mat_copy(Matrix *dst, const Matrix *src);

/* debugging */
void mat_print(const Matrix *m);

#endif