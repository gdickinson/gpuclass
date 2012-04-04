#include "rc4_serial.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>


void encrypt_stdio(char* key) {
    
    
    rc4_state_t *state = (rc4_state_t*) malloc(sizeof(rc4_state_t));
    rc4_initialize(state, key, strlen(key));
    
    
    char to_encrypt;
    char encrypted;
    
    clearerr(stdin);
    while( !feof(stdin) ) {
        to_encrypt = getchar();
        rc4_cipher(state, &to_encrypt, &encrypted, sizeof(char));
        printf("%c", encrypted);
    }
}

int main(int argc, char* argv[]) {
    char* key;
    
    if (argc != 2) {
        // Must supply a key
        fprintf(stderr, "Must supply key argument\n");
        exit(-1);
    }
    key = argv[1];
    encrypt_stdio(key);
}