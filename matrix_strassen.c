#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>

#define ADD 1
#define SUB 0

static char INFO[] = {"Strassen"};

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

bool create_matrix(Matrix *m, int row, int col)
{
    struct naive_priv *construct = malloc(sizeof(struct naive_priv));
    if ( !construct )
        return false;

    construct->values = (int **) malloc(row * sizeof(int *));
    for (int i = 0; i < col; i++)
        construct->values[i] = (int *) malloc(col * sizeof(int));
    m->priv = construct;

    return true;
}

bool seperate(const Matrix *src, Matrix *A11, Matrix *A12, Matrix *A21, Matrix *A22)
{
    int row, col, nr = 0, nc = 0, edge_r, edge_c;
    row = src->row % 2 ? src->row / 2 + 1 : src->row / 2;
    col = src->col % 2 ? src->col / 2 + 1 : src->col / 2;

    A11->row = row;
    A12->row = row;
    A21->row = row;
    A22->row = row;
    A11->col = col;
    A12->col = col;
    A21->col = col;
    A22->col = col;

    Matrix *temp = NULL;
    for (int m = 0; m < 4; m++) {

        switch(m) {
        case 0:
            create_matrix(A11, A11->row, A11->col);
            nr = 0;
            nc = 0;
            edge_r = row;
            edge_c = col;
            temp = A11;
            break;
        case 1:
            create_matrix(A12, A12->row, A12->col);
            nr = 0;
            nc = col;
            edge_r = row;
            edge_c = src->col;
            temp = A12;
            for (int i = 0 ; i < row; i++) PRIV(A12)->values[i][col - 1] = 0;
            break;
        case 2:
            create_matrix(A21, A21->row, A21->col);
            nr = row;
            nc = 0;
            edge_r = src->row;
            edge_c = col;
            temp = A21;
            for (int i = 0 ; i < col; i++) PRIV(A21)->values[row - 1][i] = 0;
            break;
        case 3:
            create_matrix(A22, A22->row, A22->col);
            nr = row;
            nc = col;
            edge_r = src->row;
            edge_c = src->col;
            temp = A22;
            for (int i = 0 ; i < col; i++) PRIV(A22)->values[row - 1][i] = 0;
            for (int i = 0 ; i < row; i++) PRIV(A22)->values[i][col - 1] = 0;
            break;
        default:
            break;
        }

        for (int r = nr; r < edge_r; r++)
            for (int c = nc; c < edge_c; c++)
                PRIV(temp)->values[r - nr][c - nc] = PRIV(src)->values[r][c];
    }
    return true;
}

bool combine(Matrix *dst, Matrix *C11, Matrix *C12, Matrix *C21, Matrix *C22)
{
    Matrix *temp = NULL;
    int nr = 0, nc = 0;
    int edge_r, edge_c;
    int row = C11->row;
    int col = C11->col;
    for (int m = 0; m < 4; m++) {
        switch(m) {
        case 0:
            temp = C11;
            nr = 0;
            nc = 0;
            edge_r = row;
            edge_c = col;
            break;
        case 1:
            temp = C12;
            nr = 0;
            nc = col;
            edge_r = row;
            edge_c = dst->col;
            break;
        case 2:
            temp = C21;
            nr = row;
            nc = 0;
            edge_r = dst->row;
            edge_c = col;
            break;
        case 3:
            temp = C22;
            nr = row;
            nc = col;
            edge_r = dst->row;
            edge_c = dst->col;
            break;
        default:
            break;
        }

        for (int r = nr; r < edge_r; r++)
            for (int c = nc; c < edge_c; c++)
                PRIV(dst)->values[r][c] = PRIV(temp)->values[r - nr][c - nc];
    }

    return true;
}

//op == ADD => add
//op == SUB => sub
bool add_sub_part(Matrix *dst, const Matrix *l, const Matrix *r, int op)
{
    if (l->col != r->col || l->row != r->row) {
        printf("can't add %dx%d matrix with %dx%d matrix!\n", l->row, l->col, r->row, r->col);
        return false;
    }

    dst->row = l->row;
    dst->col = l->col;

    if ( !dst->priv )
        create_matrix(dst, dst->row, dst->col);

    if ( op ) {
        for (int i = 0; i < dst->row; i++)
            for (int j = 0; j < dst->col; j++)
                PRIV(dst)->values[i][j] = PRIV(l)->values[i][j] + PRIV(r)->values[i][j];
    } else {
        for (int i = 0; i < dst->row; i++)
            for (int j = 0; j < dst->col; j++)
                PRIV(dst)->values[i][j] = PRIV(l)->values[i][j] - PRIV(r)->values[i][j];
    }

    return true;

}

bool mul_part(Matrix *dst, const Matrix *l, const Matrix *r)
{
    if(l->col != r->row) {
        printf("can't operate with %dx%d matrix and %dx%d matrix!\n", l->row, l->col, r->row, r->col);
        return false;
    }
    dst->row = l->row;
    dst->col = r->col;

    if ( !dst->priv )
        create_matrix(dst, dst->row, dst->col);
    else
        for (int i = 0; i < dst->row; i++)
            memset(PRIV(dst)->values[i], 0, dst->col);

    for (int i = 0; i < l->row; i++)
        for (int j = 0; j < r->col; j++)
            for (int k = 0; k < l->col; k++)
                PRIV(dst)->values[i][j] += PRIV(l)->values[i][k] *
                                           PRIV(r)->values[k][j];

    return true;
}

static bool mul(Matrix *dst, const Matrix *l, const Matrix *r)
{
    if(l->col != r->row) {
        printf("can't operate with %dx%d matrix and %dx%d matrix!\n", l->row, l->col, r->row, r->col);
        return false;
    }

    Matrix A11, A12, A21, A22;
    Matrix B11, B12, B21, B22;
    Matrix temp_l, temp_r;
    Matrix M1, M2, M3, M4, M5, M6, M7;
    Matrix C11, C12, C21, C22;

    dst->row = l->row;
    dst->col = r->col;
    temp_r.priv = NULL;
    temp_l.priv = NULL;
    M1.priv = NULL;
    M2.priv = NULL;
    M3.priv = NULL;
    M4.priv = NULL;
    M5.priv = NULL;
    M6.priv = NULL;
    M7.priv = NULL;
    C11.priv = NULL;
    C12.priv = NULL;
    C21.priv = NULL;
    C22.priv = NULL;


    create_matrix(dst, dst->row, dst->col);

    seperate(l, &A11, &A12, &A21, &A22);
    seperate(r, &B11, &B12, &B21, &B22);

    add_sub_part(&temp_l, &A11, &A22, ADD);
    add_sub_part(&temp_r, &B11, &B22, ADD);
    mul_part(&M1, &temp_l, &temp_r);

    add_sub_part(&temp_l, &A21, &A22, ADD);
    mul_part(&M2, &temp_l, &B11);

    add_sub_part(&temp_r, &B12, &B22, SUB);
    mul_part(&M3, &A11, &temp_r);

    add_sub_part(&temp_r, &B21, &B11, SUB);
    mul_part(&M4, &A22, &temp_r);

    add_sub_part(&temp_l, &A11, &A12, ADD);
    mul_part(&M5, &temp_l, &B22);

    add_sub_part(&temp_l, &A21, &A11, SUB);
    add_sub_part(&temp_r, &B11, &B12, ADD);
    mul_part(&M6, &temp_l, &temp_r);

    add_sub_part(&temp_l, &A12, &A22, SUB);
    add_sub_part(&temp_r, &B21, &B22, ADD);
    mul_part(&M7, &temp_l, &temp_r);

    add_sub_part(&temp_l, &M1, &M4, ADD);
    add_sub_part(&temp_r, &M7, &M5, SUB);
    add_sub_part(&C11, &temp_l, &temp_r, ADD);

    add_sub_part(&C12, &M3, &M5, ADD);

    add_sub_part(&C21, &M2, &M4, ADD);

    add_sub_part(&temp_l, &M1, &M2, SUB);
    add_sub_part(&temp_r, &M3, &M6, ADD);
    add_sub_part(&C22, &temp_l, &temp_r, ADD);

    combine(dst, &C11, &C12, &C21, &C22);

    return true;
}

static void get_info(char *info)
{
    strcpy(info, INFO);
}

MatrixAlgo StrassenMatrixProvider = {
    .assign = assign,
    .equal = equal,
    .mul = mul,
    .get_info = get_info,
};
