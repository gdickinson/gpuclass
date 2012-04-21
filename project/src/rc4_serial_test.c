#include "rc4_serial.h"
#include "rc4_common.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>

int main(int arglen, char** argv) {
    //Timing -www
    struct timeval t1, t2;
    
    // Allocate two identical state structs
    rc4_state_t *s1= (rc4_state_t*)malloc(sizeof(rc4_state_t));
    rc4_state_t *s2= (rc4_state_t*)malloc(sizeof(rc4_state_t));
    
    // Initialize them with the same key. Watch for the string literal
    // null-terminator which we don't bother with
    
    u_char* key = (u_char*) "Key";
    unsigned int len = (unsigned int) strlen((char*) key);
        
    rc4_initialize(s1, key, len);
    rc4_initialize(s2, key, len);
    
    // Allocate a string literal plaintext and two output buffers.
    const u_char* inputdata = (u_char*) "Plaintext";
    u_char outputdata1[10];
    u_char outputdata2[10];
    
    //Timing
    gettimeofday(&t1, 0);

    // Encipher...
    rc4_cipher(s1, inputdata, outputdata1, 10);
    
    // ...Decypher
    rc4_cipher(s2, outputdata1, outputdata2, 10);
    
    //Timing
    gettimeofday(&t2, 0);
    
    // Print the output which should be 'Plaintext'
    printf("%s\n", outputdata2);

    //Timing
    double time = (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000000.0;
    printf("Time to execute: %3.1f ms\n", time);
    
    return 0;
}
