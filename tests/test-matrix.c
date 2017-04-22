#include "matrix.h"
#include "stopwatch.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>

#define MAX 512
#define TEST_NUM 50
#define ALGO_NUM 3

static const int align = 16;

MatrixAlgo *matrix_providers[] = {
    &NaiveMatrixProvider,
    &SSEMatrixProvider,
    &StrassenMatrixProvider,
};

void generate_data(int **a, int m, int n, int range)
{
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            a[i][j] = rand() % range;
}

int main()
{
    double avg[MAX+1][ALGO_NUM], sum, log[TEST_NUM];
    char info[16];

    watch_p ctx = Stopwatch.create();
    if (!ctx)
        return -1;

    int **data = (int **) malloc(MAX * sizeof(int *));
    for (int k = 0; k < MAX; k++)
        data[k] = (int *) malloc(MAX * sizeof(int));

    FILE *fp = fopen("record.csv", "w");

    //C = A x B
    Matrix A, B, C;

    //verify multiplication
    //A x (B x r) = C x r
    Matrix r, L_Br, L_ABr, R_Cr;

    srand(time(NULL));

    for (int al = 0; al < ALGO_NUM; al++) {
        MatrixAlgo *algo = matrix_providers[al];
        algo->get_info(info);
        printf("\nalgo     : %s\n\n", info);
        for (int size = 4; size <= MAX; size *= 2) {
            sum = 0;
            for (int t = 0; t < TEST_NUM; t++) {
                algo = matrix_providers[al];
                generate_data(data, size, size, 1000);
                algo->assign(&A, size, size, (int **)data);
                generate_data(data, size, size, 1000);
                algo->assign(&B, size, size, (int **)data);
                generate_data(data, size, 1, 2);
                algo->assign(&r, size, 1, (int **)data);

                //C = A x B
                Stopwatch.reset(ctx);
                Stopwatch.start(ctx);
                if (!algo->mul(&C, &A, &B)) {
                    printf("A x B error!\n");
                    continue;
                }
                log[t] = Stopwatch.read(ctx);
                sum += log[t];
                algo = matrix_providers[0];
                //implement Freivalds’ algorithm
                if (!algo->mul(&L_Br, &B, &r)) {
                    printf("B x r error!\n");
                    continue;
                }
                if (!algo->mul(&L_ABr, &A, &L_Br)) {
                    printf("A x Br error!\n");
                    continue;
                }
                if (!algo->mul(&R_Cr, &C, &r)) {
                    printf("C x r error!\n");
                    continue;
                }
                if (!algo->equal(&L_ABr, &R_Cr))
                    printf("ABr and Cr are not equal!");
            }
            avg[size][al] = sum / TEST_NUM;
            printf("matrix   : %d x %d\n", size, size);
            printf("exe time : %lf ms\n\n", avg[size][al]);
        }
    }

    for (int size = 4; size <= MAX; size *= 2) {
        fprintf(fp, "%d", size);
        for (int al = 0; al < ALGO_NUM; al++)
            fprintf(fp, ",%lf", avg[size][al]);
        fprintf(fp, "\n");
    }

    for (int k = 0; k < MAX; k++)
        free(data[k]);
    free(data);
    fclose(fp);
    Stopwatch.destroy(ctx);
    return 0;

}
