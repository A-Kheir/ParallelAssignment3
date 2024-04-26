#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define TILE_SIZE 16

void initMatrix(float* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)rand() / RAND_MAX;
    }
}

void matrixMultiplyTiled(const float* A, const float* B, float* C, int M, int K, int N) {
    #pragma acc parallel loop collapse(2) copyin(A[0:M*K], B[0:K*N]) copyout(C[0:M*N])
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            float sum = 0.0f;
            #pragma acc loop seq
            for (int tile = 0; tile < K; tile += TILE_SIZE) {
                #pragma acc loop reduction(+:sum)
                for (int k = 0; k < TILE_SIZE; k++) {
                    int aIndex = row * K + tile + k;
                    int bIndex = (tile + k) * N + col;
                    sum += A[aIndex] * B[bIndex];
                }
            }
            C[row * N + col] = sum;
        }
    }
}

int main() {
    int rowsA = 256;
    int colsA = 256;
    int colsB = 256;

    float* A = (float*)malloc(rowsA * colsA * sizeof(float));
    float* B = (float*)malloc(colsA * colsB * sizeof(float));
    float* C = (float*)malloc(rowsA * colsB * sizeof(float));

    initMatrix(A, rowsA, colsA);
    initMatrix(B, colsA, colsB);

    clock_t start = clock();

    matrixMultiplyTiled(A, B, C, rowsA, colsA, colsB);

    clock_t stop = clock();
    double elapsedTime = ((double)(stop - start)) / (CLOCKS_PER_SEC / 1000);

    printf("Tiled matrix multiplication with OpenACC time: %.3f milliseconds\n", elapsedTime);

    free(A);
    free(B);
    free(C);

    return 0;
}
