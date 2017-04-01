#include "matrix.h"
#include <stdio.h>

MatrixAlgo *matrix_providers[] = {
    &NaiveMatrixProvider,
    &SSEMatrixProvider,
};

char info[2][8] = {"Naive", "SSE"};

int main()
{
    for (int i = 0; i <= 1; i++) {
        printf("test : %s\n", info[i]);

        MatrixAlgo *algo = matrix_providers[0];

        Matrix dst, m, n, fixed;
        algo->assign(&m, (Mat4x4) {
            .values = {
                { 1, 2, 3, 4, },
                { 5, 6, 7, 8, },
                { 1, 2, 3, 4, },
                { 5, 6, 7, 8, },
            },
        });

        algo->assign(&n, (Mat4x4) {
            .values = {
                { 1, 2, 3, 4, },
                { 5, 6, 7, 8, },
                { 1, 2, 3, 4, },
                { 5, 6, 7, 8, },
            },
        });

        algo->mul(&dst, &m, &n);

        algo->assign(&fixed, (Mat4x4) {
            .values = {
                { 34,  44,  54,  64, },
                { 82, 108, 134, 160, },
                { 34,  44,  54,  64, },
                { 82, 108, 134, 160, },
            },
        });

        if (algo->equal(&dst, &fixed))
            printf("    equal!\n\n");
        else
            printf("    not equal!\n\n");
    }

    return 0;
}
