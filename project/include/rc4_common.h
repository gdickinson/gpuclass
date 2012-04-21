#ifndef RC4_COMMON_H_5UWRSOUD
#define RC4_COMMON_H_5UWRSOUD

#include <sys/types.h>

/**
* \author Guy Dickinson <guy.dickinson@nyu.edu>
* The current state of the RC4 cipher machine
*/
typedef struct rc4_state {
    u_char permutation[256];
    u_char i;
    u_char j;
} rc4_state_t;

__BEGIN_DECLS

/**
* \author Guy Dickinson <guy.dickinson@nyu.edu>
* Utility function to exchange two arbitrary bytes in memory
*/

void swap_bytes(u_char* a, u_char* b);

/**
* \author Guy Dickinson <guy.dickinson@nyu.edu>
* Initialize a new RC4 state structure
*/

void rc4_initialize(rc4_state_t* state, const u_char* key, int keylength);

/**
\author Guy Dickinson <guy.dickinson@nyu.edu>
Returns a pointer to an array of a single constant character which is useful
for generating predictable test data
*/
u_char* initialize_constant_data(int length, u_char c);

__END_DECLS

#endif /* end of include guard: RC4_COMMON_H_5UWRSOUD */
