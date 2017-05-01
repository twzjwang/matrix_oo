#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <immintrin.h>

static char INFO[] = {"AVX"};

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
        construct->values[i] = (int *) memalign(32, thiz->col * sizeof(int));
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
        construct->values[i] = (int *) memalign(32, dst->col * sizeof(int));
    dst->priv = construct;
    for (int x = 0; x < dst->row; x += 8) {
        for (int y = 0; y < dst->col; y += 8) {
            __m256i ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7,
                    ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15;

            __m256i ymm16 = _mm256_setzero_si256();
            __m256i ymm17 = _mm256_setzero_si256();
            __m256i ymm18 = _mm256_setzero_si256();
            __m256i ymm19 = _mm256_setzero_si256();
            __m256i ymm20 = _mm256_setzero_si256();
            __m256i ymm21 = _mm256_setzero_si256();
            __m256i ymm22 = _mm256_setzero_si256();
            __m256i ymm23 = _mm256_setzero_si256();

            for (int k = 0; k < l->col; k += 8) {
                // load eight rows from source 2
                ymm0 = _mm256_load_si256((__m256i *) &(PRIV(r)->values[(k + 0)][y]));
                ymm1 = _mm256_load_si256((__m256i *) &(PRIV(r)->values[(k + 1)][y]));
                ymm2 = _mm256_load_si256((__m256i *) &(PRIV(r)->values[(k + 2)][y]));
                ymm3 = _mm256_load_si256((__m256i *) &(PRIV(r)->values[(k + 3)][y]));
                ymm4 = _mm256_load_si256((__m256i *) &(PRIV(r)->values[(k + 4)][y]));
                ymm5 = _mm256_load_si256((__m256i *) &(PRIV(r)->values[(k + 5)][y]));
                ymm6 = _mm256_load_si256((__m256i *) &(PRIV(r)->values[(k + 6)][y]));
                ymm7 = _mm256_load_si256((__m256i *) &(PRIV(r)->values[(k + 7)][y]));

                {
                    //0
                    ymm8 = _mm256_set1_epi32(PRIV(l)->values[(x + 0)][(k + 0)]);
                    ymm9 = _mm256_set1_epi32(PRIV(l)->values[(x + 0)][(k + 1)]);
                    ymm10 = _mm256_set1_epi32(PRIV(l)->values[(x + 0)][(k + 2)]);
                    ymm11 = _mm256_set1_epi32(PRIV(l)->values[(x + 0)][(k + 3)]);
                    ymm12 = _mm256_set1_epi32(PRIV(l)->values[(x + 0)][(k + 4)]);
                    ymm13 = _mm256_set1_epi32(PRIV(l)->values[(x + 0)][(k + 5)]);
                    ymm14 = _mm256_set1_epi32(PRIV(l)->values[(x + 0)][(k + 6)]);
                    ymm15 = _mm256_set1_epi32(PRIV(l)->values[(x + 0)][(k + 7)]);

                    // multiply
                    ymm8 = _mm256_mullo_epi32(ymm8, ymm0); // row 1, 2
                    ymm9 = _mm256_mullo_epi32(ymm9, ymm1);
                    ymm8 = _mm256_add_epi32(ymm8, ymm9);

                    ymm10 = _mm256_mullo_epi32(ymm10, ymm2); // row 3, 4
                    ymm11 = _mm256_mullo_epi32(ymm11, ymm3);
                    ymm10 = _mm256_add_epi32(ymm10, ymm11);

                    ymm12 = _mm256_mullo_epi32(ymm12, ymm4); // row 5, 6
                    ymm13 = _mm256_mullo_epi32(ymm13, ymm5);
                    ymm12 = _mm256_add_epi32(ymm12, ymm13);

                    ymm14 = _mm256_mullo_epi32(ymm14, ymm6); // row 7, 8
                    ymm15 = _mm256_mullo_epi32(ymm15, ymm7);
                    ymm14 = _mm256_add_epi32(ymm14, ymm15);

                    ymm8 = _mm256_add_epi32(ymm8, ymm10); // sum
                    ymm12 = _mm256_add_epi32(ymm12, ymm14);
                    ymm8 = _mm256_add_epi32(ymm8, ymm12);

                    ymm16 = _mm256_add_epi32(ymm16, ymm8);

                    //1
                    ymm8 = _mm256_set1_epi32(PRIV(l)->values[(x + 1)][(k + 0)]);
                    ymm9 = _mm256_set1_epi32(PRIV(l)->values[(x + 1)][(k + 1)]);
                    ymm10 = _mm256_set1_epi32(PRIV(l)->values[(x + 1)][(k + 2)]);
                    ymm11 = _mm256_set1_epi32(PRIV(l)->values[(x + 1)][(k + 3)]);
                    ymm12 = _mm256_set1_epi32(PRIV(l)->values[(x + 1)][(k + 4)]);
                    ymm13 = _mm256_set1_epi32(PRIV(l)->values[(x + 1)][(k + 5)]);
                    ymm14 = _mm256_set1_epi32(PRIV(l)->values[(x + 1)][(k + 6)]);
                    ymm15 = _mm256_set1_epi32(PRIV(l)->values[(x + 1)][(k + 7)]);

                    // multiply
                    ymm8 = _mm256_mullo_epi32(ymm8, ymm0); // row 1, 2
                    ymm9 = _mm256_mullo_epi32(ymm9, ymm1);
                    ymm8 = _mm256_add_epi32(ymm8, ymm9);

                    ymm10 = _mm256_mullo_epi32(ymm10, ymm2); // row 3, 4
                    ymm11 = _mm256_mullo_epi32(ymm11, ymm3);
                    ymm10 = _mm256_add_epi32(ymm10, ymm11);

                    ymm12 = _mm256_mullo_epi32(ymm12, ymm4); // row 5, 6
                    ymm13 = _mm256_mullo_epi32(ymm13, ymm5);
                    ymm12 = _mm256_add_epi32(ymm12, ymm13);

                    ymm14 = _mm256_mullo_epi32(ymm14, ymm6); // row 7, 8
                    ymm15 = _mm256_mullo_epi32(ymm15, ymm7);
                    ymm14 = _mm256_add_epi32(ymm14, ymm15);

                    ymm8 = _mm256_add_epi32(ymm8, ymm10); // sum
                    ymm12 = _mm256_add_epi32(ymm12, ymm14);
                    ymm8 = _mm256_add_epi32(ymm8, ymm12);

                    ymm17 = _mm256_add_epi32(ymm17, ymm8);

                    //2
                    ymm8 = _mm256_set1_epi32(PRIV(l)->values[(x + 2)][(k + 0)]);
                    ymm9 = _mm256_set1_epi32(PRIV(l)->values[(x + 2)][(k + 1)]);
                    ymm10 = _mm256_set1_epi32(PRIV(l)->values[(x + 2)][(k + 2)]);
                    ymm11 = _mm256_set1_epi32(PRIV(l)->values[(x + 2)][(k + 3)]);
                    ymm12 = _mm256_set1_epi32(PRIV(l)->values[(x + 2)][(k + 4)]);
                    ymm13 = _mm256_set1_epi32(PRIV(l)->values[(x + 2)][(k + 5)]);
                    ymm14 = _mm256_set1_epi32(PRIV(l)->values[(x + 2)][(k + 6)]);
                    ymm15 = _mm256_set1_epi32(PRIV(l)->values[(x + 2)][(k + 7)]);

                    // multiply
                    ymm8 = _mm256_mullo_epi32(ymm8, ymm0); // row 1, 2
                    ymm9 = _mm256_mullo_epi32(ymm9, ymm1);
                    ymm8 = _mm256_add_epi32(ymm8, ymm9);

                    ymm10 = _mm256_mullo_epi32(ymm10, ymm2); // row 3, 4
                    ymm11 = _mm256_mullo_epi32(ymm11, ymm3);
                    ymm10 = _mm256_add_epi32(ymm10, ymm11);

                    ymm12 = _mm256_mullo_epi32(ymm12, ymm4); // row 5, 6
                    ymm13 = _mm256_mullo_epi32(ymm13, ymm5);
                    ymm12 = _mm256_add_epi32(ymm12, ymm13);

                    ymm14 = _mm256_mullo_epi32(ymm14, ymm6); // row 7, 8
                    ymm15 = _mm256_mullo_epi32(ymm15, ymm7);
                    ymm14 = _mm256_add_epi32(ymm14, ymm15);

                    ymm8 = _mm256_add_epi32(ymm8, ymm10); // sum
                    ymm12 = _mm256_add_epi32(ymm12, ymm14);
                    ymm8 = _mm256_add_epi32(ymm8, ymm12);

                    ymm18 = _mm256_add_epi32(ymm18, ymm8);

                    //3
                    ymm8 = _mm256_set1_epi32(PRIV(l)->values[(x + 3)][(k + 0)]);
                    ymm9 = _mm256_set1_epi32(PRIV(l)->values[(x + 3)][(k + 1)]);
                    ymm10 = _mm256_set1_epi32(PRIV(l)->values[(x + 3)][(k + 2)]);
                    ymm11 = _mm256_set1_epi32(PRIV(l)->values[(x + 3)][(k + 3)]);
                    ymm12 = _mm256_set1_epi32(PRIV(l)->values[(x + 3)][(k + 4)]);
                    ymm13 = _mm256_set1_epi32(PRIV(l)->values[(x + 3)][(k + 5)]);
                    ymm14 = _mm256_set1_epi32(PRIV(l)->values[(x + 3)][(k + 6)]);
                    ymm15 = _mm256_set1_epi32(PRIV(l)->values[(x + 3)][(k + 7)]);

                    // multiply
                    ymm8 = _mm256_mullo_epi32(ymm8, ymm0); // row 1, 2
                    ymm9 = _mm256_mullo_epi32(ymm9, ymm1);
                    ymm8 = _mm256_add_epi32(ymm8, ymm9);

                    ymm10 = _mm256_mullo_epi32(ymm10, ymm2); // row 3, 4
                    ymm11 = _mm256_mullo_epi32(ymm11, ymm3);
                    ymm10 = _mm256_add_epi32(ymm10, ymm11);

                    ymm12 = _mm256_mullo_epi32(ymm12, ymm4); // row 5, 6
                    ymm13 = _mm256_mullo_epi32(ymm13, ymm5);
                    ymm12 = _mm256_add_epi32(ymm12, ymm13);

                    ymm14 = _mm256_mullo_epi32(ymm14, ymm6); // row 7, 8
                    ymm15 = _mm256_mullo_epi32(ymm15, ymm7);
                    ymm14 = _mm256_add_epi32(ymm14, ymm15);

                    ymm8 = _mm256_add_epi32(ymm8, ymm10); // sum
                    ymm12 = _mm256_add_epi32(ymm12, ymm14);
                    ymm8 = _mm256_add_epi32(ymm8, ymm12);

                    ymm19 = _mm256_add_epi32(ymm19, ymm8);

                    //4
                    ymm8 = _mm256_set1_epi32(PRIV(l)->values[(x + 4)][(k + 0)]);
                    ymm9 = _mm256_set1_epi32(PRIV(l)->values[(x + 4)][(k + 1)]);
                    ymm10 = _mm256_set1_epi32(PRIV(l)->values[(x + 4)][(k + 2)]);
                    ymm11 = _mm256_set1_epi32(PRIV(l)->values[(x + 4)][(k + 3)]);
                    ymm12 = _mm256_set1_epi32(PRIV(l)->values[(x + 4)][(k + 4)]);
                    ymm13 = _mm256_set1_epi32(PRIV(l)->values[(x + 4)][(k + 5)]);
                    ymm14 = _mm256_set1_epi32(PRIV(l)->values[(x + 4)][(k + 6)]);
                    ymm15 = _mm256_set1_epi32(PRIV(l)->values[(x + 4)][(k + 7)]);

                    // multiply
                    ymm8 = _mm256_mullo_epi32(ymm8, ymm0); // row 1, 2
                    ymm9 = _mm256_mullo_epi32(ymm9, ymm1);
                    ymm8 = _mm256_add_epi32(ymm8, ymm9);

                    ymm10 = _mm256_mullo_epi32(ymm10, ymm2); // row 3, 4
                    ymm11 = _mm256_mullo_epi32(ymm11, ymm3);
                    ymm10 = _mm256_add_epi32(ymm10, ymm11);

                    ymm12 = _mm256_mullo_epi32(ymm12, ymm4); // row 5, 6
                    ymm13 = _mm256_mullo_epi32(ymm13, ymm5);
                    ymm12 = _mm256_add_epi32(ymm12, ymm13);

                    ymm14 = _mm256_mullo_epi32(ymm14, ymm6); // row 7, 8
                    ymm15 = _mm256_mullo_epi32(ymm15, ymm7);
                    ymm14 = _mm256_add_epi32(ymm14, ymm15);

                    ymm8 = _mm256_add_epi32(ymm8, ymm10); // sum
                    ymm12 = _mm256_add_epi32(ymm12, ymm14);
                    ymm8 = _mm256_add_epi32(ymm8, ymm12);

                    ymm20 = _mm256_add_epi32(ymm20, ymm8);

                    //5
                    ymm8 = _mm256_set1_epi32(PRIV(l)->values[(x + 5)][(k + 0)]);
                    ymm9 = _mm256_set1_epi32(PRIV(l)->values[(x + 5)][(k + 1)]);
                    ymm10 = _mm256_set1_epi32(PRIV(l)->values[(x + 5)][(k + 2)]);
                    ymm11 = _mm256_set1_epi32(PRIV(l)->values[(x + 5)][(k + 3)]);
                    ymm12 = _mm256_set1_epi32(PRIV(l)->values[(x + 5)][(k + 4)]);
                    ymm13 = _mm256_set1_epi32(PRIV(l)->values[(x + 5)][(k + 5)]);
                    ymm14 = _mm256_set1_epi32(PRIV(l)->values[(x + 5)][(k + 6)]);
                    ymm15 = _mm256_set1_epi32(PRIV(l)->values[(x + 5)][(k + 7)]);

                    // multiply
                    ymm8 = _mm256_mullo_epi32(ymm8, ymm0); // row 1, 2
                    ymm9 = _mm256_mullo_epi32(ymm9, ymm1);
                    ymm8 = _mm256_add_epi32(ymm8, ymm9);

                    ymm10 = _mm256_mullo_epi32(ymm10, ymm2); // row 3, 4
                    ymm11 = _mm256_mullo_epi32(ymm11, ymm3);
                    ymm10 = _mm256_add_epi32(ymm10, ymm11);

                    ymm12 = _mm256_mullo_epi32(ymm12, ymm4); // row 5, 6
                    ymm13 = _mm256_mullo_epi32(ymm13, ymm5);
                    ymm12 = _mm256_add_epi32(ymm12, ymm13);

                    ymm14 = _mm256_mullo_epi32(ymm14, ymm6); // row 7, 8
                    ymm15 = _mm256_mullo_epi32(ymm15, ymm7);
                    ymm14 = _mm256_add_epi32(ymm14, ymm15);

                    ymm8 = _mm256_add_epi32(ymm8, ymm10); // sum
                    ymm12 = _mm256_add_epi32(ymm12, ymm14);
                    ymm8 = _mm256_add_epi32(ymm8, ymm12);

                    ymm21 = _mm256_add_epi32(ymm21, ymm8);

                    //6
                    ymm8 = _mm256_set1_epi32(PRIV(l)->values[(x + 6)][(k + 0)]);
                    ymm9 = _mm256_set1_epi32(PRIV(l)->values[(x + 6)][(k + 1)]);
                    ymm10 = _mm256_set1_epi32(PRIV(l)->values[(x + 6)][(k + 2)]);
                    ymm11 = _mm256_set1_epi32(PRIV(l)->values[(x + 6)][(k + 3)]);
                    ymm12 = _mm256_set1_epi32(PRIV(l)->values[(x + 6)][(k + 4)]);
                    ymm13 = _mm256_set1_epi32(PRIV(l)->values[(x + 6)][(k + 5)]);
                    ymm14 = _mm256_set1_epi32(PRIV(l)->values[(x + 6)][(k + 6)]);
                    ymm15 = _mm256_set1_epi32(PRIV(l)->values[(x + 6)][(k + 7)]);

                    // multiply
                    ymm8 = _mm256_mullo_epi32(ymm8, ymm0); // row 1, 2
                    ymm9 = _mm256_mullo_epi32(ymm9, ymm1);
                    ymm8 = _mm256_add_epi32(ymm8, ymm9);

                    ymm10 = _mm256_mullo_epi32(ymm10, ymm2); // row 3, 4
                    ymm11 = _mm256_mullo_epi32(ymm11, ymm3);
                    ymm10 = _mm256_add_epi32(ymm10, ymm11);

                    ymm12 = _mm256_mullo_epi32(ymm12, ymm4); // row 5, 6
                    ymm13 = _mm256_mullo_epi32(ymm13, ymm5);
                    ymm12 = _mm256_add_epi32(ymm12, ymm13);

                    ymm14 = _mm256_mullo_epi32(ymm14, ymm6); // row 7, 8
                    ymm15 = _mm256_mullo_epi32(ymm15, ymm7);
                    ymm14 = _mm256_add_epi32(ymm14, ymm15);

                    ymm8 = _mm256_add_epi32(ymm8, ymm10); // sum
                    ymm12 = _mm256_add_epi32(ymm12, ymm14);
                    ymm8 = _mm256_add_epi32(ymm8, ymm12);

                    ymm22 = _mm256_add_epi32(ymm22, ymm8);

                    //7
                    ymm8 = _mm256_set1_epi32(PRIV(l)->values[(x + 7)][(k + 0)]);
                    ymm9 = _mm256_set1_epi32(PRIV(l)->values[(x + 7)][(k + 1)]);
                    ymm10 = _mm256_set1_epi32(PRIV(l)->values[(x + 7)][(k + 2)]);
                    ymm11 = _mm256_set1_epi32(PRIV(l)->values[(x + 7)][(k + 3)]);
                    ymm12 = _mm256_set1_epi32(PRIV(l)->values[(x + 7)][(k + 4)]);
                    ymm13 = _mm256_set1_epi32(PRIV(l)->values[(x + 7)][(k + 5)]);
                    ymm14 = _mm256_set1_epi32(PRIV(l)->values[(x + 7)][(k + 6)]);
                    ymm15 = _mm256_set1_epi32(PRIV(l)->values[(x + 7)][(k + 7)]);

                    // multiply
                    ymm8 = _mm256_mullo_epi32(ymm8, ymm0); // row 1, 2
                    ymm9 = _mm256_mullo_epi32(ymm9, ymm1);
                    ymm8 = _mm256_add_epi32(ymm8, ymm9);

                    ymm10 = _mm256_mullo_epi32(ymm10, ymm2); // row 3, 4
                    ymm11 = _mm256_mullo_epi32(ymm11, ymm3);
                    ymm10 = _mm256_add_epi32(ymm10, ymm11);

                    ymm12 = _mm256_mullo_epi32(ymm12, ymm4); // row 5, 6
                    ymm13 = _mm256_mullo_epi32(ymm13, ymm5);
                    ymm12 = _mm256_add_epi32(ymm12, ymm13);

                    ymm14 = _mm256_mullo_epi32(ymm14, ymm6); // row 7, 8
                    ymm15 = _mm256_mullo_epi32(ymm15, ymm7);
                    ymm14 = _mm256_add_epi32(ymm14, ymm15);

                    ymm8 = _mm256_add_epi32(ymm8, ymm10); // sum
                    ymm12 = _mm256_add_epi32(ymm12, ymm14);
                    ymm8 = _mm256_add_epi32(ymm8, ymm12);

                    ymm23 = _mm256_add_epi32(ymm23, ymm8);
                }
            }
            _mm256_store_si256((__m256i *)(&PRIV(dst)->values[(x + 0)][y]), ymm16);
            _mm256_store_si256((__m256i *)(&PRIV(dst)->values[(x + 1)][y]), ymm17);
            _mm256_store_si256((__m256i *)(&PRIV(dst)->values[(x + 2)][y]), ymm18);
            _mm256_store_si256((__m256i *)(&PRIV(dst)->values[(x + 3)][y]), ymm19);
            _mm256_store_si256((__m256i *)(&PRIV(dst)->values[(x + 4)][y]), ymm20);
            _mm256_store_si256((__m256i *)(&PRIV(dst)->values[(x + 5)][y]), ymm21);
            _mm256_store_si256((__m256i *)(&PRIV(dst)->values[(x + 6)][y]), ymm22);
            _mm256_store_si256((__m256i *)(&PRIV(dst)->values[(x + 7)][y]), ymm23);
        }
    }

    return true;
}

static void get_info(char *info)
{
    strcpy(info, INFO);
}

MatrixAlgo AVXMatrixProvider = {
    .assign = assign,
    .equal = equal,
    .mul = mul,
    .get_info = get_info,
};
