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

__global__ void findMaxNaivelyKernel(int* A) {
    unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    for (unsigned int step = 1; step < (blockDim.x * gridDim.x); step *= 2) {
        __syncthreads();
        if (tid % (2 * step) == 0) {
            int candidate = A[tid + step];
            if (A[tid] < candidate) {
                A[tid] = candidate;
            }
        }
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
    
    dim3 dimBlock(512);
    dim3 dimGrid(length/512, 1);
    
    findMaxNaivelyKernel<<<dimGrid, dimBlock>>>(cudaArray);
    
    
    // Recover just the first element from the device to save time.
    cudaMemcpy(&ret, cudaArray, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(cudaArray);
    return ret;
}

__global__ void findMaxWithoutDivergenceKernel(int* A) {
    unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    for (unsigned int step = blockDim.x * gridDim.x >> 1; step > 0; step >>= 1) {
        __syncthreads();
        if (tid < step) {
            int candidate = A[tid + step];
            if (A[tid] < candidate) {
                A[tid] = candidate;
            }
        }
    }
}

int cudaFindMaxWithoutDivergence(int* A, int length) {
    int size = length * sizeof(int);
    int ret;
    int* cudaArray;
    cudaMalloc(&cudaArray, size);
    cudaMemcpy(cudaArray, A, size, cudaMemcpyHostToDevice);
    
    dim3 dimBlock(512);
    dim3 dimGrid(length / 512, 1);
    
    findMaxWithoutDivergenceKernel<<<dimGrid, dimBlock>>>(cudaArray);
    
    // Recover just the first element from the device to save time.
    cudaMemcpy(&ret, cudaArray, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(cudaArray);
    return ret;
}

__global__ void findMaxWithSharedMemoryKernel(int* A) {
    //Static allocation like this is pretty lame but good enough for testing.
    __shared__ int sharedmem[512];
    
    // copy this chunk into shared memory from global
    unsigned int threadIndex = threadIdx.x;
    unsigned int globalThreadId = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int nThreads = (blockDim.x * gridDim.x);
    
    sharedmem[threadIndex] = (globalThreadId < nThreads) ?
        A[globalThreadId] : 0;
        
    
    // Do the reduction in shared memory
    for (unsigned int step = blockDim.x >> 1; step > 0; step >>= 1) {
        __syncthreads();
        if (threadIndex < step) {
            int candidate = sharedmem[threadIndex + step];
            if (sharedmem[threadIndex] < candidate) {
                sharedmem[threadIndex] = candidate;
            }
        }
    }
    __syncthreads();
    
    // Thread 0 within a block writes the result back to global memory
    if (threadIndex == 0) {
        A[blockIdx.x] = sharedmem[0];
    }
    __syncthreads();
    
    // Now the global data structure has as elements 0-gridDim.x filled with
    // the results which we need to reduce again. 
    if (globalThreadId < gridDim.x) {
        for (unsigned int step = gridDim.x >> 1; step > 0; step >>= 1) {
            int candidate = A[globalThreadId + step];
            if (A[globalThreadId] < candidate) {
                A[globalThreadId] = candidate;
            }
        }
    }
}

int cudaFindMaxWithSharedMemory(int* A, int length) {
    int size = length * sizeof(int);
    int ret;
    int* cudaArray;
    cudaMalloc(&cudaArray, size);
    cudaMemcpy(cudaArray, A, size, cudaMemcpyHostToDevice);
    
    dim3 dimBlock(512);
    dim3 dimGrid(length / 512, 1);
    
    findMaxWithSharedMemoryKernel<<<dimGrid, dimBlock>>>(cudaArray);
    
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

void printTiming(timeval start, timeval end, char* desc, int length) {
    double elapsed =
        (end.tv_usec - start.tv_usec);
    printf("%s length %d: %g usec\n", desc, elapsed, length);
}

void checkAndPrintResult(int expected, int actual, char* desc) {
    if (expected == actual) {
        printf("%s passed check ", desc);
    } else {
        printf("%s failed check " , desc);
    }
    printf("(expected %d, got %d)\n", expected, actual);
}

void launchTestWithTiming(
        int testType,
        int length) {
            
    int actual;
    char* desc;
    
    timeval start, end;
    int* array = initializeArray(length);
    int expected = array[length - 1];
    gettimeofday(&start, NULL);
    switch (testType) {
        // CPU
        case 0:
        desc = "Serial";
        actual = findMaxOnHost(array, length);
        break;
        
        // Naive Parallel
        case 1:
        desc = "Naive Parallel";
        actual = cudaFindMaxOnDeviceNaively(array, length);
        break;
        
        // Non Divergent Parallel
        case 2:
        desc = "Non Divergent Parallel";
        actual = cudaFindMaxWithoutDivergence(array, length);
        break;
        
        // Shared Memory Parallel
        case 3:
        desc = "Shared Memory Parallel";
        actual = cudaFindMaxWithSharedMemory(array, length);
        break;
        
        default:
        desc = "Unrecognized!";
        break;
        
    }
    gettimeofday(&end, NULL);
    free(array);
    
    checkAndPrintResult(expected, actual, desc);
    printTiming(start, end, desc, length);
    
}


void runTest(int length) {
    launchTestWithTiming(0, length);
    launchTestWithTiming(1, length);
    launchTestWithTiming(1, length);
    launchTestWithTiming(2, length);
    launchTestWithTiming(3, length);
    }

int main(void) {
    runTest(1024);
    runTest(4096);
    return 0;
}