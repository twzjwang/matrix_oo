#ifndef MATRIX_H_
#define MATRIX_H_

#include <stdbool.h>

typedef struct {
    int row, col;
    void *priv;
} Matrix;

typedef struct {
    void (*assign)(Matrix *thiz, int row, int col, int **data);
    bool (*equal)(const Matrix *l, const Matrix *r);
    bool (*mul)(Matrix *dst, const Matrix *l, const Matrix *r);
} MatrixAlgo;

/* Available matrix providers */
extern MatrixAlgo NaiveMatrixProvider;
extern MatrixAlgo SSEMatrixProvider;

#endif /* MATRIX_H_ */
