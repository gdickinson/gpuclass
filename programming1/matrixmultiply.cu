#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <sys/time.h>

/*
* hostmultiply.cu
* Copyright 2012 Guy Dickinson <guy.dickinson@nyu.edu>
*
* Written for "GPUs: Architecture and Programming"
* Prof. M. Zahran, New York University
*
* Derived in part from code in "Programming Massively Parallel Processors: 
* A Hands-On Approach" by David Kirk and Wen-mei Hwu.
*/


// Vanilla matrix multiplication on the host
void matrixMulOnHost(float* M, float* N, float* P, int width) {   
    for (int i = 0; i < width; ++i)
        for (int j = 0; j < width; ++j) {
            double sum = 0;
            for (int k = 0; k < width; ++k) {
                double a = M[i * width + k];
                double b = N[k * width + j];
                sum += a * b;
            }
    P[i * width + j] = sum;
    }
}

// Matrix Multiplication Kernel
__global__ void MatrixMulKernel(float* Md, float* Nd, float* Pd, int width) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    float pvalue = 0;
    
    for (int k = 0; k < width; ++k) {
        float Mdelement = Md[ty * width + k];
        float Ndelement = Nd[ty * width + k];
        pvalue += Mdelement * Ndelement;
    }
    Pd[ty * width + tx] = pvalue;
}

// M and N are matrices to be multiplied
// P is the result
void cudaMatrixMul(float* M, float* N, float* P, int width) {
    
    int size = width * width * sizeof(float);
    
    float* Md;
    float* Nd;
    float* Pd;
    
    // Transfer M and N to device memory
    cudaMalloc(&Md, size);
    cudaMemcpy(Md, M, size, cudaMemcpyHostToDevice);
    
    cudaMalloc(&Nd, size);
    cudaMemcpy(Nd, N, size, cudaMemcpyHostToDevice);
    
    // Allocate P on the device
    cudaMalloc(&Pd, size);
    
    // Invocation
    dim3 dimBlock(width, width);
    dim3 dimGrid(1,1);
    
    MatrixMulKernel<<<dimGrid, dimBlock>>>(Md, Nd, Pd, width);
    
    // Transfer P from device to host
    cudaMemcpy(P, Pd, size, cudaMemcpyDeviceToHost);
    
    // Free device matrices
    cudaFree(Md);
    cudaFree(Pd);
    cudaFree(Nd);
}

void runTest(void) {
    int doublings = 8;
    int widths[doublings];
    for (int i = 1; i <= doublings; i++) {
        widths[i - 1] = pow(2, i + 2);
    }
    
    for (int i = 0; i < doublings; i ++) {
        
        int width = widths[i];
        int size = width * width * sizeof(float);
        
        timeval serialStart, serialEnd;
        timeval parallelStart, parallelEnd;
        double serialElapsedTime;
        double parallelElapsedTime;
        
        float *m;
        float *n;
        float *p;
        
        m = (float*) malloc(size);
        n = (float*) malloc(size);
        p = (float*) malloc(size);
        
        for (int j = 0; j < width * width; j++) {
            n[j] = 0.0;
            m[j] = 1.0;
        }
        
        
        gettimeofday(&serialStart, NULL);
        matrixMulOnHost(m, n, p, width);
        gettimeofday(&serialEnd, NULL);
        
        gettimeofday(&parallelStart, NULL);
        cudaMatrixMul(m, n, p, width);
        gettimeofday(&parallelEnd, NULL);
        
        serialElapsedTime = 
            (serialEnd.tv_sec - serialStart.tv_sec) * 1000.0;
        serialElapsedTime += 
            (serialEnd.tv_usec - serialStart.tv_usec) / 1000.0;
        
        parallelElapsedTime = 
            (parallelEnd.tv_sec - parallelStart.tv_sec) * 1000.0;
        parallelElapsedTime += 
            (parallelEnd.tv_usec - parallelStart.tv_usec) / 1000.0;
        
        double speedup = (serialElapsedTime / parallelElapsedTime) * 100.0;
        printf("%d x %d: Serial: %f\t\tParallel %f\t(%f%% Speedup)\n",
            width, width, serialElapsedTime, parallelElapsedTime, speedup);
        
        free(m);
        free(n);
    }
}

int main(void) {
    runTest();
    }
