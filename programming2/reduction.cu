#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

/*
TODO: Head Comment

Copyright 2012 Guy Dickinson <guy.dickinson@nyu.edu>
*/

// Vanilla, sequential reduction on host
// This *would* have a divergence problem if it were multithreaded
int findMaxOnHost(int* A, int length) {
    for (int step = 1; step < length; step *= 2) {
        for (int i = 0; i < length; i += (2 * step)) {
            int candidate = A[i + step];
            if (A[i] < candidate) {
                A[i] = candidate;
            }
        }
    }
    return A[0];
}

__global__ void findMaxNaivelyKernel(int* A, int length) {
    unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    for (unsigned int step = 1; step < blockDim.x; step *= 2) {
        __syncthreads();
        if (tid % (2 * step) == 0) {
            int candidate = A[tid + step];
            if (A[tid] < candidate) {
                A[tid] = candidate;
            }
        }
        __syncthreads();
    }
    
}

// Naively find the maximum element, without taking into account thread
// diversion or memory efficiency
int cudaFindMaxOnDeviceNaively(int* A, int length) {
    int size = length * sizeof(int);
    int ret;
    int* cudaArray;
    cudaMalloc(&cudaArray, size);
    cudaMemcpy(cudaArray, A, size, cudaMemcpyHostToDevice);
    
    dim3 dimBlock(1024 % length);
    dim3 dimGrid(length / 1024, 1);
    
    findMaxNaivelyKernel<<<dimGrid, dimBlock>>>(cudaArray, length);
    
    // Recover just the first element from the device to save time.
    cudaMemcpy(&ret, cudaArray, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(cudaArray);
    return ret;
}




int* initializeArray(int length) {
    // Set up an array of ints of the right length
    void *ptr;
    ptr = malloc(length * sizeof(int));
    if (ptr == NULL) {
        // Handle allocation error
    }
    int* A = (int*) ptr;
    
    // Fill it with ints
    int j = 1;
    for (int i = 0; i < length; i++) {
        A[i] = j;
        j += 2;
    }
    return A;
}


void runTest(int length) {
    timeval serialStart, serialEnd;
    int* serialArr = initializeArray(length);
    int* naiveParallelArray = initializeArray(length);
    
    gettimeofday(&serialStart, NULL);
    int serialMax = findMaxOnHost(serialArr, length);
    gettimeofday(&serialEnd, NULL);
    free(serialArr);
    
    int expectedParallelMax = naiveParallelArray[length - 1];
    int naiveParallelMax =
        cudaFindMaxOnDeviceNaively(naiveParallelArray, length);
    
    
    
    // Make sure we actually found the max value
    if (serialMax == serialArr[length - 1]) {
        printf("Serial reduction passed check (expected %d, got %d)\n",
            serialMax, serialArr[length - 1]);
    } else {
        printf("Serial reduction failed! (expected %d, got %d)\n",
            serialMax, serialArr[length - 1]);
    }
    
    if (expectedParallelMax == naiveParallelArray[length - 1]) {
        printf("Naive Parallel reduction passed check (expected %d, got %d)\n",
            expectedParallelMax, naiveParallelArray[length - 1]);
    } else {
        printf("Naive Parallel reduction failed! (expected %d, got %d)\n",
            expectedParallelMax, naiveParallelArray[length - 1]);
    }
    
    double serialElapsedTime =
        (serialEnd.tv_sec - serialStart.tv_sec) * 1000.0;
    printf("Serial time: %g\n", serialElapsedTime); 
}

int main(void) {
    runTest(1024);
    return 0;
}