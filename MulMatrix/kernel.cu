#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#ifndef __CUDACC__ 
#define __CUDACC__
#endif
#include <device_functions.h>
#include <stdio.h>
#include <time.h>
#include <cstdlib>
#include <math.h>

const int BLOCK_SIZE = 32;
const int N = 256;
const int RANDOM_NUMBER = 101;

__global__ void mulMatrixDevice(int* C, int* A, int* B, int N){
	int blockX = blockIdx.x;
	int	blockY = blockIdx.y;
	int threadX = threadIdx.x;
	int threadY = threadIdx.y;
	int aBegin = N * BLOCK_SIZE * blockY;
	int bBegin = BLOCK_SIZE * blockX;
	int aEnd = aBegin + N - 1;
	int aStep = BLOCK_SIZE;
	int	bStep = BLOCK_SIZE * N;
	int sum = 0;
	for (int indexA = aBegin, indexB = bBegin; indexA <= aEnd; indexA += aStep, indexB += bStep){
		__shared__ int As[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ int Bs[BLOCK_SIZE][BLOCK_SIZE];
		As[threadY][threadX] = A[indexA + N * threadY + threadX];
		Bs[threadY][threadX] = B[indexB + N * threadY + threadX];
		__syncthreads();
		for (int k = 0; k < BLOCK_SIZE; k++)
			sum += As[threadY][k] * Bs[k][threadX];
		__syncthreads();
	}
	int c = N * BLOCK_SIZE * blockY + BLOCK_SIZE * blockX;
	C[c + N * threadY + threadX] = sum;
}

void mulMatrixHost(int* a, int *b, int* c, int n) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			int sum = 0;
			for (int k = 0; k < n; k++)
				sum = sum + a[i * n + k] * b[k * n + j];
			c[i * n + j] = sum;
		}
	}
}

int* randomMatrix(int* matrix, int n) {
	srand(time(NULL));
	for (int i = 0; i < n; ++i) {
		matrix[i] = rand() % RANDOM_NUMBER - round(RANDOM_NUMBER/2);
	}
	return matrix;
}

bool checkMatrix(int* matrix1, int* matrix2, int size) {
	for (int i = 0; i < size; ++i) {
		if (matrix1[i] != matrix2[i])
			return false;
	}
	return true;
}

int main(){
	int matrixSize = N*N;
	int* matrixA = new int[matrixSize];
	int* matrixB = new int[matrixSize];
	int* hostMatrixC = new int[matrixSize];
	int* deviceMatrixC = new int[matrixSize];

	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_c = 0;
	cudaMalloc((void**)&dev_a, matrixSize * sizeof(int));
	cudaMalloc((void**)&dev_b, matrixSize * sizeof(int));
	cudaMalloc((void**)&dev_c, matrixSize * sizeof(int));
	randomMatrix(matrixA, matrixSize);
	randomMatrix(matrixB, matrixSize);

	clock_t time = clock();
	mulMatrixHost(matrixA, matrixB, hostMatrixC, N);
	double hostTime = double(clock() - time) * 1000 / CLOCKS_PER_SEC;

	cudaEvent_t start, stop;
	float deviceTime = 0.0f;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	cudaMemcpy(dev_a, matrixA, matrixSize * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, matrixB, matrixSize * sizeof(int), cudaMemcpyHostToDevice);
	mulMatrixDevice << <dim3(N / BLOCK_SIZE, N / BLOCK_SIZE), dim3(BLOCK_SIZE, BLOCK_SIZE) >> >(dev_c, dev_a, dev_b, N);
	cudaDeviceSynchronize();
	cudaMemcpy(deviceMatrixC, dev_c, matrixSize * sizeof(int), cudaMemcpyDeviceToHost);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&deviceTime, start, stop);

	if (checkMatrix(hostMatrixC, deviceMatrixC, matrixSize)) {
		printf("CPU: %f\n", hostTime);
		printf("GPU: %f\n", deviceTime);
	}
	else {
		printf("Matrixs not equals!!!");
	}
	free(matrixA);
	free(matrixB);
	free(deviceMatrixC);
	free(hostMatrixC);
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	system("pause");
	return 0;
}


