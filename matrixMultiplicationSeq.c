#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SIZE 1024

int a[SIZE][SIZE];
int b[SIZE][SIZE];
long int c[SIZE][SIZE];

void init_matrices() {
    int i, j;
    for (i = 0; i < SIZE; i++) {
        for (j = 0; j < SIZE; j++) {
            a[i][j] = (i + 1) / 2;
            b[i][j] = (i + 1) / 2;
            c[i][j] = 0;
        }
    }
}

void multiply_matrices() {
    int i, j, k;
    for (i = 0; i < SIZE; i++) {
        for (j = 0; j < SIZE; j++) {
            for (k = 0; k < SIZE; k++) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}

int main() {
    clock_t start, end;
    double cpu_time_used;
    
    init_matrices();
    
    start = clock();
    
    multiply_matrices();
    
    end = clock();
    
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    
    printf("Multiplication SEQUENTIAL TIME took  %lf seconds\n", cpu_time_used);
    
    return 0;
}