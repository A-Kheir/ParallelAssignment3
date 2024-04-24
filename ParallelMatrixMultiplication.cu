%%cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void matrixMultiplication(float* MatA, float* MatB, float* MatC, int Size) {
  int Row = blockIdx.y * blockDim.y + threadIdx.y;
  int Col = blockIdx.x * blockDim.x + threadIdx.x;

  if ((Row < Size) && (Col < Size)) {
    float Cvalue = 0;
    for (int k = 0; k < Size; ++k) {
      Cvalue += MatA[Row * Size + k] * MatB[k * Size + Col];
    }
    MatC[Row * Size + Col] = Cvalue;
  }
}

__global__ void matrixRectangule(float* MatA, float* MatB, float* MatC, int WidthA, int HeightA, int WidthB, int HeightB) {
  int Row = blockIdx.y * blockDim.y + threadIdx.y;
  int Col = blockIdx.x * blockDim.x + threadIdx.x;

  if ((Row < HeightA) && (Col < WidthB)) {
    float Cvalue = 0;
    for (int k = 0; k < WidthA; ++k) {
      Cvalue += MatA[Row * WidthA + k] * MatB[k * WidthB + Col];
    }
    MatC[Row * WidthB + Col] = Cvalue;
  }
}

int main() {
  int WidthA = 800;
  int HeightA = 800;
  int WidthB = 800;
  int HeightB = 800;

  size_t sizeA = WidthA * HeightA * sizeof(float);
  size_t sizeB = WidthB * HeightB * sizeof(float);
  size_t sizeC;

  float *host_A, *host_B, *host_C;
  host_A = (float*)malloc(sizeA);
  host_B = (float*)malloc(sizeB);
  host_C = nullptr;

  float *device_A, *device_B, *device_C;
  cudaMalloc((void**)&device_A, sizeA);
  cudaMalloc((void**)&device_B, sizeB);

  cudaMemcpy(device_A, host_A, sizeA, cudaMemcpyHostToDevice);
  cudaMemcpy(device_B, host_B, sizeB, cudaMemcpyHostToDevice);

  dim3 blockDim(16, 16);
  dim3 gridDim;

  if (WidthA == HeightA && WidthB == HeightB) {
    sizeC = WidthA * HeightB * sizeof(float);
    host_C = (float*)malloc(sizeC);
    cudaMalloc((void**)&device_C, sizeC);
    gridDim = dim3((WidthA + blockDim.x - 1) / blockDim.x, (HeightB + blockDim.y - 1) / blockDim.y);
    matrixMultiplication<<<gridDim, blockDim>>>(device_A, device_B, device_C, WidthA);
    cudaMemcpy(host_C, device_C, sizeC, cudaMemcpyDeviceToHost);
    cudaFree(device_C);
  } else {
    sizeC = HeightA * WidthB * sizeof(float);
    host_C = (float*)malloc(sizeC);
    cudaMalloc((void**)&device_C, sizeC);
    gridDim = dim3((WidthB + blockDim.x - 1) / blockDim.x, (HeightA + blockDim.y - 1) / blockDim.y);
    matrixRectangule<<<gridDim, blockDim>>>(device_A, device_B, device_C, WidthA, HeightA, WidthB, HeightB);
    cudaMemcpy(host_C, device_C, sizeC, cudaMemcpyDeviceToHost);
    cudaFree(device_C);
  }

  clock_t start, stop;
  start = clock();

  if (WidthA == HeightA && WidthB == HeightB) {
    matrixMultiplication<<<gridDim, blockDim>>>(device_A, device_B, device_C, WidthA);
  } else {
    matrixRectangule<<<gridDim, blockDim>>>(device_A, device_B, device_C, WidthA, HeightA, WidthB, HeightB);
  }

  cudaDeviceSynchronize();
  stop = clock();
  float milliseconds = ((float)(stop - start) / CLOCKS_PER_SEC) * 1000.0;
  printf("Parallel Execution Time (No Tiling): %.4f seconds\n", milliseconds / 1000.0);

  free(host_A);
  free(host_B);
  free(host_C);

  cudaFree(device_A);
  cudaFree(device_B);

  return 0;
}
