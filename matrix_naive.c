#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>

struct naive_priv {
    int **values;
};

#define PRIV(x) \
    ((struct naive_priv *) ((x)->priv))

static void assign(Matrix *thiz, int row, int col, int **data)
{
    thiz->row = row;
    thiz->col = col;

    struct naive_priv *construct = malloc(sizeof(struct naive_priv));
    construct->values = (int **) malloc(thiz->row * sizeof(int *));
    for (int i = 0; i < thiz->row; i++)
        construct->values[i] = (int *) malloc(thiz->col * sizeof(int));
    thiz->priv = construct;

    for (int i = 0; i < thiz->row; i++)
        for (int j = 0; j < thiz->col; j++)
            PRIV(thiz)->values[i][j] = data[i][j];
}

static const float epsilon = 1 / 10000.0;

static bool equal(const Matrix *l, const Matrix *r)
{
    if (l->row != r->row || l->col != r->col) {
        printf("%dx%d matrix and %dx%d matrix are not equal!\n", l->row, l->col, r->row, r->col);
        return false;
    }
    for (int i = 0; i < l->row; i++)
        for (int j = 0; j < l->col; j++)
            if (PRIV(l)->values[i][j] + epsilon < PRIV(r)->values[i][j] ||
                    PRIV(r)->values[i][j] + epsilon < PRIV(l)->values[i][j])
                return false;
    return true;
}

static bool mul(Matrix *dst, const Matrix *l, const Matrix *r)
{
    if(l->col != r->row) {
        printf("can't operate with %dx%d matrix and %dx%d matrix!\n", l->row, l->col, r->row, r->col);
        return false;
    }
    dst->row = l->row;
    dst->col = r->col;

    struct naive_priv *construct = malloc(sizeof(struct naive_priv));
    if (!construct)
        return false;

    construct->values = (int **) malloc(dst->row * sizeof(int *));
    for (int i = 0; i < dst->row; i++)
        construct->values[i] = (int *) malloc(dst->col * sizeof(int));
    dst->priv = construct;

    for (int i = 0; i < l->row; i++)
        for (int j = 0; j < r->col; j++)
            for (int k = 0; k < l->col; k++)
                PRIV(dst)->values[i][j] += PRIV(l)->values[i][k] *
                                           PRIV(r)->values[k][j];
    return true;
}

MatrixAlgo NaiveMatrixProvider = {
    .assign = assign,
    .equal = equal,
    .mul = mul,
};
