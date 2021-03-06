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

#include "rc4_common.h"
#include "rc4_parallel.cuh"
#include <cuda.h>
#include <stdio.h>


int main (int arglen, char** argv) {

    //Timing - wwward
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    u_char* data = initialize_constant_data(1024, 'a');
    rc4_state_t *state = (rc4_state_t*)malloc(sizeof(rc4_state_t));
    
    u_char* initkey = (u_char*) "key";
    
    rc4_initialize(state, initkey, 3);
    
    
    u_char key[1024];
    
    get_n_bytes_of_key(state, key, 1024);
    
    
    u_char* cudaData;
    u_char* cudaKey;
    
    cudaMalloc(&cudaData, 1024);
    cudaMemcpy(cudaData, data, 1024, cudaMemcpyHostToDevice);
    cudaMalloc(&cudaKey, 1024);
    cudaMemcpy(cudaKey, key, 1024, cudaMemcpyHostToDevice);
    
	//Timing
    cudaEventRecord(start,0);
    rc4_crypt_kernel<<<2,512>>>(cudaData, cudaKey, 1024);
	//Timing
    cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);

    printf("Kernel launch error: %s\n", cudaGetErrorString(cudaGetLastError()));
    
    cudaMemcpy(data, cudaData, 1024, cudaMemcpyDeviceToHost);
    data[1023] = '\0';
    printf("data: %s\n", data);

	//Timing
	cudaEventElapsedTime(&time,start,stop);
	printf("Time for kernel: %f ms\n",time);
    
}