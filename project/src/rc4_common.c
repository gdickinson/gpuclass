#include <sys/types.h>
#include <stdlib.h>
#include "rc4_common.h"


u_char* initialize_constant_data(int length, u_char c) {
    u_char* data = (u_char*) malloc(sizeof(u_char) * length);
    int i;
    for (i = 0; i < length; i++) {
        data[i] = c;
    }
    return data;
}

void swap_bytes(u_char *a, u_char *b) {
    u_char temp;
    temp = *a;
    *a = *b;
    *b = temp;
}

void rc4_initialize(rc4_state_t *state, const u_char *key, int keylength) {
    int i;
    u_char j;
    
    for (i = 0; i<256; i++) {
        state->permutation[i] = i;
    }
    state->i = 0;
    state->j = 0;
    
    for (i = j = 0; i<256; i++) {
        j+= state->permutation[i] + key[ i % keylength];
        swap_bytes(&state->permutation[i], &state->permutation[j]);
    }
}