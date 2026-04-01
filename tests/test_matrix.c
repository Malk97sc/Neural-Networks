#include "matrix.h"

int main(){
    int rows, cols;
    Matrix A, B;
    
    rows = cols = 3;

    A = mat_alloc(rows, cols);

    mat_fill(&A, 1.5f);
    mat_print(&A);

    B = mat_alloc(rows, cols);
    mat_copy(&B, &A);

    mat_print(&B);

    mat_free(&A);
    mat_free(&B);

    return 0;
}