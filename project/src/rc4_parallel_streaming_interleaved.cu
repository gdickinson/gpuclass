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

#include "rc4_parallel.cuh"
#include "rc4_common.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <cuda.h>

// Utility function to setup and return a pointer to a new state
rc4_state_t* setup_state_with_key(u_char* key, int keylen) {
	rc4_state_t* s = (rc4_state_t*) malloc(sizeof(rc4_state_t));
	rc4_initialize(s, key, keylen);
	return s;
	
}

void print_buffer_in_hex(u_char* buf, int size) {
	int i;
	for (i=0; i<size; i++) {
		printf("0x%x ", buf[i] );
	}
	printf("\n");
}

void allocate_host_buffers(u_char* bufarr[], int size, int num) {
	int cnt;
	cudaError_t err;
	for(cnt = 0; cnt < num; cnt++) {
		err = cudaMallocHost((void**) &bufarr[cnt], (size * sizeof(u_char)));
		if (err) {
			fprintf(stderr, 
				"Error allocating host buffer: %s\n",
				cudaGetErrorString(err));
		}
	}
}


void allocate_device_buffers(u_char* bufarr[], int size, int num) {
	int cnt;
	cudaError_t err;
	for(cnt = 0; cnt < num; cnt++) {
		err = cudaMalloc(&bufarr[cnt], (size * sizeof(u_char)));
		if (err) {
			fprintf(stderr, 
				"Error allocating host buffer: %s\n",
				cudaGetErrorString(err));
		}
	}
}

void launch_kernel(u_char* data, u_char* key, int size) {
	int nblocks;
	int nthreads;
	
	nblocks = (size / 512);
	if (nblocks == 0) {
		nblocks = 1;
	}
	
	nthreads = 512;
	rc4_crypt_kernel<<<nblocks, nthreads>>>(data, key, size);
}


void encrypt_stdin_buffered_parallel(int bufsize, rc4_state* s) {
	
	int bufcnt = 3;
	
	u_char* host_databuffers[bufcnt];
	u_char* device_databuffers[bufcnt];
	u_char* host_keybuffers[bufcnt];
	u_char* device_keybuffers[bufcnt];
	
	// Allocate buffers on both host and device
	allocate_host_buffers(host_databuffers, bufsize, bufcnt);
	allocate_host_buffers(host_keybuffers, bufsize, bufcnt);
	allocate_device_buffers(device_databuffers, bufsize, bufcnt);
	allocate_device_buffers(device_keybuffers, bufsize, bufcnt);
	
	
	// CUDA Streams (one for copy, one for compute)
	cudaStream_t copyStream;
	cudaStream_t computeStream;
	cudaStreamCreate(&copyStream);
	cudaStreamCreate(&computeStream);
	

	size_t sz = sizeof(u_char);	
	
	// State flags
	int bufnum = 0;
	int capturedbytes = 0;
	int bytesininputbuf = 0;
	int bytescopyingtodevice = 0;
	int bytesreadytorun = 0;
	int bytestoreadback = 0;
	int bytestoprint = 0;
	int eof = 0;
	int alldone = 0;
	
	// Workloop
	while (!alldone){
		
		if (bytesininputbuf > 0) {
			// Copy data to device
			// data
			cudaMemcpyAsync(device_databuffers[(bufnum+2)%3],
				host_databuffers[(bufnum + 2) % 3],
				bytesininputbuf,
				cudaMemcpyHostToDevice,
				copyStream);
			// key
			cudaMemcpyAsync(device_keybuffers[(bufnum+2)%3],
				host_keybuffers[(bufnum + 2) % 3],
				bytesininputbuf,
				cudaMemcpyHostToDevice,
				copyStream);
			
			bytescopyingtodevice = bytesininputbuf;
			bytesininputbuf = 0;
			} else {
				bytescopyingtodevice = 0;
			}
			
		
		if (bytestoreadback > 0) {
			// Get computed data from device
			cudaMemcpyAsync(host_databuffers[(bufnum+1)%3],
							device_databuffers[bufnum],
							bytestoreadback,
							cudaMemcpyDeviceToHost,
							copyStream);
		
			bytestoprint = bytestoreadback;
			bytestoreadback = 0;
		}
		
		
		// Launch kernel
		if (bytesreadytorun > 0) {
			
			launch_kernel(device_databuffers[(bufnum+1)%3],
			 	device_keybuffers[(bufnum+1)%3],
				bytesreadytorun);
		
			bytestoreadback = bytesreadytorun;
			
		}
		bytesreadytorun = bytescopyingtodevice;
		
		
		// Capture new data
		while (capturedbytes < bufsize) {
			if (feof(stdin)) {
				eof = 1;
				break;
			}
			capturedbytes += fread(
				&host_databuffers[bufnum][capturedbytes],
				sz,
				(bufsize - capturedbytes),
				stdin);
		}
		
		get_n_bytes_of_key(s,
			host_keybuffers[bufnum],
			capturedbytes);
		bytesininputbuf = capturedbytes;
		capturedbytes = 0;
				
		
		cudaDeviceSynchronize();
		
		// Print the bytes we just read back
		if (bytestoprint) {			
			fwrite(host_databuffers[(bufnum + 1) % 3],
			1,
			bytestoprint,
			stdout);
			
			bytestoprint = 0;
		}
		
		
		// Housekeeping
		bufnum++;
		bufnum %= 3;
		
		if (eof && !bytesininputbuf 
			&& !bytescopyingtodevice
			&& !bytesreadytorun
			&& !bytestoreadback
			&& !bytestoprint) {alldone = 1;}
					
	}
	
	// Free buffers
	int j;
	for (j = 0; j < bufcnt; j++) {
		cudaFree(device_keybuffers[j]);
		cudaFreeHost(host_keybuffers[j]);
		cudaFree(device_databuffers[j]);
		cudaFreeHost(host_databuffers[j]);
	}
}



int main(int argc, char *argv[]) {
	if (argc == 1) {
		printf("Must specify key as arg 1\n");
		exit(255);
	}

    int buffersz = 512; //size shipped to gpu

    if (argc == 3) {
        buffersz = atoi(argv[2]);
    }
	
	int keylen = strlen(argv[1]);
	
	u_char* key = (u_char*) malloc(keylen);
	memcpy(key, argv[1], keylen);
	rc4_state_t* state = setup_state_with_key(key, keylen);
	encrypt_stdin_buffered_parallel(buffersz, state);
	free(state);	
}

