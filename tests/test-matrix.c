#include "matrix.h"
#include "stopwatch.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#define MAX 512
#define TEST 25

MatrixAlgo *matrix_providers[] = {
    &NaiveMatrixProvider,
    &SSEMatrixProvider,
};

int main()
{
    Matrix dst, m, n, fixed;
    double exe_time, sum[2];
    char info[16];
    watch_p ctx = Stopwatch.create();
    if (!ctx)
        return -1;

    srand(time(NULL));

    FILE *fp = fopen("record.csv", "w");

    int **data_n = (int **) malloc(MAX * sizeof(int *));
    for (int k = 0; k < MAX; k++)
        data_n[k] = (int *) malloc(MAX * sizeof(int));

    int **data_m = (int **) malloc(MAX * sizeof(int *));
    for (int k = 0; k < MAX; k++)
        data_m[k] = (int *) malloc(MAX * sizeof(int));

    int **data_fixed = (int **) malloc(MAX * sizeof(int *));
    for (int k = 0; k < MAX; k++)
        data_fixed[k] = (int *) malloc(MAX * sizeof(int));

    for (int j = 1; j <= MAX; j *= 2) {
        for (int i = 0; i <= 1; i++)
            sum[i] = 0;
        for (int t = 0; t < TEST; t++) {
            for (int k = 0; k < j; k++)
                for (int l = 0; l < j; l++)
                    data_n[k][l] = rand() % 100;

            for (int k = 0; k < j; k++)
                for (int l = 0; l < j; l++)
                    data_m[k][l] = rand() % 100;

            for (int k = 0; k < j; k++)
                for (int l = 0; l < j; l++) {
                    data_fixed[k][l] = 0;
                    for(int p = 0; p < j; p++)
                        data_fixed[k][l] += data_n[k][p] * data_m[p][l];
                }

            for (int i = 0; i <= 1; i++) {
                MatrixAlgo *algo = matrix_providers[i];
                if (i == 1 && j % 4 != 0)
                    continue;

                algo->assign(&m, j, j, data_n);
                algo->assign(&n, j, j, data_m);
                algo->assign(&fixed, j, j, data_fixed);

                Stopwatch.reset(ctx);
                Stopwatch.start(ctx);
                int f = algo->mul(&dst, &m, &n);
                exe_time = Stopwatch.read(ctx);
                sum[i] += exe_time;

                if (!f)
                    printf("result   : mul error!\n");
                else if (!algo->equal(&dst, &fixed))
                    printf("result   : not equal!\n");
            }
        }
        fprintf(fp, "%d ", j);
        for (int i = 0; i <= 1; i++) {
            MatrixAlgo *algo = matrix_providers[i];
            algo->get_info(info);
            printf("\ntest     : %s\n", info);
            printf("matrix   : %d x %d\n", j, j);
            printf("exe time : %lf ms\n", sum[i] / TEST);
            fprintf(fp, ",%lf", sum[i] / TEST);
        }
        fprintf(fp, "\n");
    }

    for (int k = 0; k < MAX; k++) {
        free(data_n[k]);
        free(data_m[k]);
        free(data_fixed[k]);
    }

    free(data_n);
    free(data_m);
    free(data_fixed);
    Stopwatch.destroy(ctx);
    fclose(fp);
    return 0;
}
