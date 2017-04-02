#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>

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

bool sse_mul(Matrix *dst, const Matrix *l, const Matrix *r)
{
    if (l->col != r->row) {
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

    return true;
}

MatrixAlgo SSEMatrixProvider = {
    .assign = assign,
    .equal = equal,
    .mul = sse_mul,
};
