#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>

#define ADD 1
#define SUB 0

static char INFO[] = {"Strassen SSE"};

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

static bool create_matrix(Matrix *m, int row, int col)
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

static bool seperate(const Matrix *src, Matrix *A11, Matrix *A12, Matrix *A21, Matrix *A22)
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

static bool combine(Matrix *dst, Matrix *C11, Matrix *C12, Matrix *C21, Matrix *C22)
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
static bool add_sub_part(Matrix *dst, const Matrix *l, const Matrix *r, int op)
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

static bool mul_part(Matrix *dst, const Matrix *l, const Matrix *r)
{
    if(l->col != r->row) {
        printf("can't operate with %dx%d matrix and %dx%d matrix!\n", l->row, l->col, r->row, r->col);
        return false;
    }
    dst->row = l->row;
    dst->col = r->col;

    if ( !dst->priv )
        create_matrix(dst, dst->row, dst->col);

    for (int x = 0; x < dst->row; x += 4) {
        for (int y = 0; y < dst->col; y += 4) {
            __m128i des0 = _mm_setzero_si128 ();
            __m128i des1 = _mm_setzero_si128 ();
            __m128i des2 = _mm_setzero_si128 ();
            __m128i des3 = _mm_setzero_si128 ();

            for (int k = 0; k < l->col; k += 4) {
                __m128i I0 = _mm_load_si128((__m128i *)&(PRIV(l)->values[(x + 0)][k]));
                __m128i I1 = _mm_load_si128((__m128i *)&(PRIV(l)->values[(x + 1)][k]));
                __m128i I2 = _mm_load_si128((__m128i *)&(PRIV(l)->values[(x + 2)][k]));
                __m128i I3 = _mm_load_si128((__m128i *)&(PRIV(l)->values[(x + 3)][k]));

                __m128i I4 = _mm_set_epi32 (PRIV(r)->values[(k+3)][y], PRIV(r)->values[(k+2)][y],
                                            PRIV(r)->values[(k+1)][y], PRIV(r)->values[k][y]);
                __m128i I5 = _mm_set_epi32 (PRIV(r)->values[(k+3)][(y+1)],
                                            PRIV(r)->values[(k+2)][(y+1)], PRIV(r)->values[(k+1)][(y+1)],
                                            PRIV(r)->values[(k+0)][(y+1)]);
                __m128i I6 = _mm_set_epi32 (PRIV(r)->values[(k+3)][(y+2)],
                                            PRIV(r)->values[(k+2)][(y+2)], PRIV(r)->values[(k+1)][(y+2)],
                                            PRIV(r)->values[(k+0)][(y+2)]);
                __m128i I7 = _mm_set_epi32 (PRIV(r)->values[(k+3)][(y+3)],
                                            PRIV(r)->values[(k+2)][(y+3)], PRIV(r)->values[(k+1)][(y+3)],
                                            PRIV(r)->values[(k+0)][(y+3)]);

                __m128i T0 = _mm_mullo_epi32(I0, I4);
                __m128i T1 = _mm_mullo_epi32(I0, I5);
                __m128i T2 = _mm_mullo_epi32(I0, I6);
                __m128i T3 = _mm_mullo_epi32(I0, I7);

                __m128i T4 = _mm_mullo_epi32(I1, I4);
                __m128i T5 = _mm_mullo_epi32(I1, I5);
                __m128i T6 = _mm_mullo_epi32(I1, I6);
                __m128i T7 = _mm_mullo_epi32(I1, I7);

                __m128i T8 = _mm_mullo_epi32(I2, I4);
                __m128i T9 = _mm_mullo_epi32(I2, I5);
                __m128i T10 = _mm_mullo_epi32(I2, I6);
                __m128i T11 = _mm_mullo_epi32(I2, I7);

                __m128i T12 = _mm_mullo_epi32(I3, I4);
                __m128i T13 = _mm_mullo_epi32(I3, I5);
                __m128i T14 = _mm_mullo_epi32(I3, I6);
                __m128i T15 = _mm_mullo_epi32(I3, I7);

                __m128i T16 = _mm_unpacklo_epi32(T0, T1);
                __m128i T17 = _mm_unpacklo_epi32(T2, T3);
                __m128i T18 = _mm_unpackhi_epi32(T0, T1);
                __m128i T19 = _mm_unpackhi_epi32(T2, T3);

                __m128i T20 = _mm_unpacklo_epi64(T16, T17);
                __m128i T21 = _mm_unpackhi_epi64(T16, T17);
                __m128i T22 = _mm_unpacklo_epi64(T18, T19);
                __m128i T23 = _mm_unpackhi_epi64(T18, T19);

                T20 = _mm_add_epi32(T20, T21);
                T20 = _mm_add_epi32(T20, T22);
                T20 = _mm_add_epi32(T20, T23);

                des0 = _mm_add_epi32(T20, des0);

                T16 = _mm_unpacklo_epi32(T4, T5);
                T17 = _mm_unpacklo_epi32(T6, T7);
                T18 = _mm_unpackhi_epi32(T4, T5);
                T19 = _mm_unpackhi_epi32(T6, T7);

                T20 = _mm_unpacklo_epi64(T16, T17);
                T21 = _mm_unpackhi_epi64(T16, T17);
                T22 = _mm_unpacklo_epi64(T18, T19);
                T23 = _mm_unpackhi_epi64(T18, T19);

                T20 = _mm_add_epi32(T20, T21);
                T20 = _mm_add_epi32(T20, T22);
                T20 = _mm_add_epi32(T20, T23);

                des1 = _mm_add_epi32(T20, des1);

                T16 = _mm_unpacklo_epi32(T8, T9);
                T17 = _mm_unpacklo_epi32(T10, T11);
                T18 = _mm_unpackhi_epi32(T8, T9);
                T19 = _mm_unpackhi_epi32(T10, T11);

                T20 = _mm_unpacklo_epi64(T16, T17);
                T21 = _mm_unpackhi_epi64(T16, T17);
                T22 = _mm_unpacklo_epi64(T18, T19);
                T23 = _mm_unpackhi_epi64(T18, T19);

                T20 = _mm_add_epi32(T20, T21);
                T20 = _mm_add_epi32(T20, T22);
                T20 = _mm_add_epi32(T20, T23);

                des2 = _mm_add_epi32(T20, des2);

                T16 = _mm_unpacklo_epi32(T12, T13);
                T17 = _mm_unpacklo_epi32(T14, T15);
                T18 = _mm_unpackhi_epi32(T12, T13);
                T19 = _mm_unpackhi_epi32(T14, T15);

                T20 = _mm_unpacklo_epi64(T16, T17);
                T21 = _mm_unpackhi_epi64(T16, T17);
                T22 = _mm_unpacklo_epi64(T18, T19);
                T23 = _mm_unpackhi_epi64(T18, T19);

                T20 = _mm_add_epi32(T20, T21);
                T20 = _mm_add_epi32(T20, T22);
                T20 = _mm_add_epi32(T20, T23);

                des3 = _mm_add_epi32(T20, des3);
            }

            _mm_store_si128((__m128i *)(&PRIV(dst)->values[(x + 0)][y]), des0);
            _mm_store_si128((__m128i *)(&PRIV(dst)->values[(x + 1)][y]), des1);
            _mm_store_si128((__m128i *)(&PRIV(dst)->values[(x + 2)][y]), des2);
            _mm_store_si128((__m128i *)(&PRIV(dst)->values[(x + 3)][y]), des3);
        }
    }

    /*
    for (int i = 0; i < l->row; i++)
        for (int j = 0; j < r->col; j++)
            for (int k = 0; k < l->col; k++)
                PRIV(dst)->values[i][j] += PRIV(l)->values[i][k] *
                                           PRIV(r)->values[k][j];
    */

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

MatrixAlgo StrassenSSEMatrixProvider = {
    .assign = assign,
    .equal = equal,
    .mul = mul,
    .get_info = get_info,
};
