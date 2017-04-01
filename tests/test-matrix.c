#include "matrix.h"
#include "stopwatch.h"
#include <stdio.h>
#include <unistd.h>

MatrixAlgo *matrix_providers[] = {
    &NaiveMatrixProvider,
    &SSEMatrixProvider,
};

char info[2][8] = {"Naive", "SSE"};

int main()
{
    double time;

    watch_p ctx = Stopwatch.create();
    if (!ctx) return -1;

    for (int i = 0; i <= 1; i++) {
        printf("test    : %s\n", info[i]);

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

        Stopwatch.reset(ctx);
        Stopwatch.start(ctx);

        algo->mul(&dst, &m, &n);

        time = Stopwatch.read(ctx);

        algo->assign(&fixed, (Mat4x4) {
            .values = {
                { 34,  44,  54,  64, },
                { 82, 108, 134, 160, },
                { 34,  44,  54,  64, },
                { 82, 108, 134, 160, },
            },
        });

        if (algo->equal(&dst, &fixed))
            printf("result   : equal!\n");
        else
            printf("result   : not equal!\n");
        printf("exe time : %lf ms\n\n", time);
    }

    Stopwatch.destroy(ctx);
    return 0;
}
