#ifndef MATRIX_H_
#define MATRIX_H_

#include <stdbool.h>
#include <string.h>

typedef struct {
    int row, col;
    void *priv;
} Matrix;

typedef struct {
    void (*assign)(Matrix *thiz, int row, int col, int **data);
    bool (*equal)(const Matrix *l, const Matrix *r);
    bool (*mul)(Matrix *dst, const Matrix *l, const Matrix *r);
    void (*get_info)(char *info);
} MatrixAlgo;

/* Available matrix providers */
extern MatrixAlgo NaiveMatrixProvider;
extern MatrixAlgo SSEMatrixProvider;
extern MatrixAlgo StrassenMatrixProvider;

#endif /* MATRIX_H_ */
