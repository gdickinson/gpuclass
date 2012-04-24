#include "rc4_parallel.cuh"
#include "rc4_common.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <cuda.h>

void print_buffer_in_hex(u_char* buf, int size) {
	int i;
	for (i=0; i<size; i++) {
		printf("0x%x ", buf[i] );
	}
	printf("\n");
}

void encrypt_chunk(u_char* host_databuf, u_char* host_keybuf, 
	u_char* cuda_databuf, u_char* cuda_keybuf, int length) {		

	cudaMemcpy(cuda_databuf, host_databuf, length, cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_keybuf, host_keybuf, length, cudaMemcpyHostToDevice);
	
	int threads = 256;
	int blocks = (length / 256);
	if (blocks == 0) {
		blocks = 1;
	}
	
	
	rc4_crypt_kernel<<<blocks, threads>>>(cuda_databuf, cuda_keybuf, length);
	
	fprintf(stderr, "Kernel launch error? %s\n", cudaGetErrorString(cudaGetLastError()));
	
	if (host_databuf[length] != '\0') {
		fprintf(stderr, "host_databuf isn't null terminated\n");
	}
	
	// Recover encrypted material
	cudaMemcpy(host_databuf, cuda_databuf, length, cudaMemcpyDeviceToHost);
	fwrite(host_databuf, sizeof(u_char), length, stdout);
	
}


void encrypt_stdin_buffered_parallel(int bufsize, rc4_state* s) {
	
	u_char* buffer = (u_char*) malloc(sizeof(u_char) * (bufsize) );
	
	// Null-terminate the buffer for printing; we never write to the null-byte
	
	u_char* keybuffer = (u_char*) malloc(sizeof(u_char) * bufsize);
	u_char* cuda_keybuffer;
	u_char* cuda_databuffer;
	// u_char* output = (u_char*) malloc(sizeof(u_char) * bufsize);
		
	cudaMalloc(&cuda_keybuffer, bufsize);
	cudaMalloc(&cuda_databuffer, bufsize);


	size_t sz = sizeof(u_char);	
	
	int i = 0;
	while (!feof(stdin)){
		while (i < bufsize && !feof(stdin)) {
			int j = fread( &buffer[i], sz, (bufsize - i), stdin);
			i += j;
		}
		
		
		get_n_bytes_of_key(s, keybuffer, i);
		
		encrypt_chunk(buffer, keybuffer, cuda_databuffer, cuda_keybuffer, i);	
		
		i = 0;
	}
	//printf("Reached EOF\n");
	free(buffer);
	free(keybuffer);
}

rc4_state_t* setup_state_with_key(u_char* key, int keylen) {
	rc4_state_t* s = (rc4_state_t*) malloc(sizeof(rc4_state_t));
	rc4_initialize(s, key, keylen);
	return s;
	
}

int main(int argc, char *argv[]) {
	if (argc == 1) {
		printf("Must specify key as arg 1\n");
		exit(255);
	}
	//printf("Key is %s\n", argv[1]);
	
	int keylen = strlen(argv[1]);
	//printf("Keylen is %d\n", keylen);
	
    int buffersz = 512; //size shipped to gpu

    if (argc == 3) {
        buffersz = atoi(argv[2]);
    }

	u_char* key = (u_char*) malloc(keylen);
	memcpy(key, argv[1], keylen);
	rc4_state_t* state = setup_state_with_key(key, keylen);
	encrypt_stdin_buffered_parallel(buffersz, state);	
}

