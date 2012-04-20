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

__global__ void rc4_crypt_kernel(u_char* data, u_char* key, int length) {
    
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    data[tid] = data[tid] ^ key[tid];
    
}

void get_n_bytes_of_key(rc4_state_t* state, u_char* outbuf, int n) {
    int i;
    for (i = 0; i < n; i++) {
        u_char j;
        
        state->i++;
        state->j += state->permutation[state->i];
        
        swap_bytes(&state->permutation[state->i],
            &state->permutation[state->j]);
        
        j = state->permutation[state->i] + state->permutation[state->j];
        
        outbuf[i] = state->permutation[j];
    }
}