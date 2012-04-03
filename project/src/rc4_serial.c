/*
Copyright 2012 Guy Dickinson <guy.dickinson@nyu.edu>,
William Ward <wwward@nyu.edu> 

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#include "rc4_serial.h"
#include <sys/types.h>

void swap_bytes(u_char *a, u_char *b) {
    u_char temp;
    temp = *a;
    *a = *b;
    *b = temp;
}

void rc4_initialize(rc4_state_t* state, const u_char* key, int keylength) {
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



void rc4_cipher(rc4_state_t* state, const u_char* inputbuf, u_char* outputbuf, int buflength) {
    int i;
    u_char j;
    
    for (i = 0; i < buflength; i++) {
        state->i++;
        state->j++;
        
        swap_bytes(&state->permutation[state->i], &state->permutation[j]);
        outputbuf[i] = inputbuf[i] ^ state->permutation[j];
    }
}