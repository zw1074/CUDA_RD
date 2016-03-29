/*
   This file using RD alogrithm to solve tridiagonal matrix.
   Author: Zihao Wang
   Email: zw1074@nyu.edu
*/

#include <stdio.h>
#include "math.h"
#include "time.h"

#define Width 3
#define Height 3
#define Depth (1<<25)

__global__ void TimesMatrixPara(float *B, float *C, int a) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= a && k < Depth) {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                C[k*3*3 + i*3 + j] = 0;
                for (int t = 0; t < 3; t++) {
                    C[k*3*3 + i*3 + j] += B[k*3*3 + i*3 + t] * B[(k - a)*3*3 + t*3 + j];                
                }
            }
        }
    }
}

__global__ void Transfer(float *B, float *C) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < Depth) {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                B[k*9 + i*3 + j] = C[k*9 + i*3 + j];
            }
        }
    }
}

__global__ void Scalar(float *B, float *x, float *X, int N) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= 1 && k < N-1) {
        X[k] = B[k*9]*x[0] + B[k*9 + 1]*x[1] + B[k*9 + 2]*x[2];
    }
    if (k == N - 1) {
        X[k] = B[k*9 + 3]*x[0] + B[k*9 + 4]*x[1] + B[k*9 + 5]*x[2];
    }
}

int main() {
    clock_t start, finish;
    float costtime;
    float *B;
    B = (float*)malloc(sizeof(float)*Depth*Width*Height);

    printf("The size is %d\n", Depth);
    //Initialize B
    for (int k = 0;k < Depth; k++) {
        if (k == Depth - 1) {
            B[k*9] = -2.0f;
        }
        else {
            B[k*9] = 2.0f;
        }
        if (k == 0 || k == Depth - 1) {
            B[k*9 + 1] = 1.0f;
            if (k == 0) {
                B[k*9 + 2] =-1.0f;
            }
            else {
                B[k*9 + 2] = 1.0f;
            }
        }
        else {
            B[k*9 + 1] = -1.0f;
            B[k*9 + 2] = 0.0f;
        }  
        B[k*9 + 3] = 1.0f;
        B[k*9 + 4] = 0.0f;
        B[k*9 + 5] = 0.0f;
        B[k*9 + 6] = 0.0f;
        B[k*9 + 7] = 0.0f;
        B[k*9 + 8] = 1.0f;
    }
    float *d_B, *d_C;
    cudaMalloc(&d_B, sizeof(float)*Depth*Width*Height);
    cudaMalloc(&d_C, sizeof(float)*Depth*Width*Height);
    start = clock();

    // Allocate the assignment
    cudaMemcpy(d_B, B, sizeof(float)*Depth*Width*Height, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, B, sizeof(float)*Depth*Width*Height, cudaMemcpyHostToDevice);

    // Calculate B on each cuda_core
    for (int i = 0; i < 20; i++) {
        TimesMatrixPara<<<(Depth + 1023)/1024, 1024>>>(d_B, d_C, 1<<i);
        Transfer<<<(Depth + 1023)/1024, 1024>>>(d_B,d_C);
    }
    
    // Calculate each scalar
    cudaMemcpy(B, d_B, sizeof(float)*Depth*Width*Height, cudaMemcpyDeviceToHost);
    // Calculate first scalar
    float *X;
    X = (float*)malloc(sizeof(float)*Depth);
    X[0] = -B[(Depth - 1)*9 + 2]/B[(Depth - 1)*9];
    float *d_X;
    cudaMalloc(&d_X, sizeof(float)*Depth);
    cudaMemcpy(d_X, X, sizeof(float)*Depth, cudaMemcpyHostToDevice);
    float *x;
    x = (float*)malloc(sizeof(float)*3);
    float *d_x;
    cudaMalloc(&d_x, sizeof(float)*3);
    x[0] = X[0];
    x[1] = 0;
    x[2] = 1;
    cudaMemcpy(d_x, x, sizeof(float)*3, cudaMemcpyHostToDevice);
    Scalar<<<(Depth + 1023)/1024, 1024>>>(d_B, d_x, d_X, Depth);
    cudaMemcpy(X, d_X, sizeof(float)*Depth, cudaMemcpyDeviceToHost);
    finish = clock();
    costtime = (float) (finish - start)/ CLOCKS_PER_SEC;
    float maxError = 0.0f;
    for (int i = 0; i < Depth; i++) maxError = max(maxError, abs(X[i] - 1.0f));
    printf("maxError = %f, Time Cost = %f\n", maxError, costtime);
    return 0;
}
