#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

/*
TODO: Head Comment

Copyright 2012 Guy Dickinson <guy.dickinson@nyu.edu>
*/

// Vanilla, sequential reduction on host
// length must be a multiple of 2.
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
    int* arr = initializeArray(length);
    
    gettimeofday(&serialStart, NULL);
    int serialMax = findMaxOnHost(arr, length);
    gettimeofday(&serialEnd, NULL);
    
    // Make sure we actually found the max value
    if (serialMax == arr[length - 1]) {
        printf("Serial reduction passed check\n");
    } else {
        printf("Serial reduction failed!\n");
    }
    
    double serialElapsedTime =
        (serialEnd.tv_sec - serialStart.tv_sec) * 1000.0;
    printf("Serial time: %g\n", serialElapsedTime); 
}

int main(void) {
    runTest(4096);
    return 0;
}