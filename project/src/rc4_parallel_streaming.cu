#include "rc4_parallel.cuh"
#include "rc4_common.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>

void encrypt_stdin_buffered_parallel(int bufsize, rc4_state* s) {
	
	u_char* buffer = (u_char*) malloc(sizeof(u_char) * bufsize);
	
	size_t sz = sizeof(u_char);
	
	int i = 0;
	while (!feof(stdin)){
		while (i < bufsize && !feof(stdin)) {
			int j = fread( &buffer[i], sz, (bufsize - i), stdin);
			i += j;
			printf ("Got %d bytes, buffer now %d bytes long\n", j, i );	
		}
		
		// Launch a kernel here
		
		printf("Do work on %d bytes\n", i);
		i = 0;
	}
	printf("Reached EOF\n");
	free(buffer);
}

rc4_state_t* setup_state_with_key(u_char* key, int keylen) {
	rc4_state_t* s = (rc4_state_t*) malloc(sizeof(rc4_state_t));
	rc4_initialize(s, key, keylen);
	return s;
	
}

int main(int argc, char* argv) {
	
	rc4_state_t* state = setup_state_with_key((u_char*) argv[1], strlen(argv[1])-1);
	
	
}

