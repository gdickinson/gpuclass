#include "rc4_serial.h"
#include <stdlib.h>
#include <stdio.h>

int main(char* argv) {
    
    // Allocate two identical state structs
    rc4_state_t *s1= (rc4_state_t*)malloc(sizeof(rc4_state_t));
    rc4_state_t *s2= (rc4_state_t*)malloc(sizeof(rc4_state_t));
    
    // Initialize them with the same key. Watch for the string literal
    // null-terminator which we don't bother with
    rc4_initialize(s1, "Key", 3);
    rc4_initialize(s2, "Key", 3);
    
    // Allocate a string literal plaintext and two output buffers.
    const char* inputdata = "Plaintext";
    char outputdata1[10];
    char outputdata2[10];
    
    // Encipher...
    rc4_cipher(s1, inputdata, outputdata1, 10);
    
    // ...Decypher
    rc4_cipher(s2, outputdata1, outputdata2, 10);
    
    // Print the output which should be 'Plaintext'
    printf("%s\n", outputdata2);
    
    return 0;
}