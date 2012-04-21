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

#include <sys/types.h>
#include "rc4_common.h"

__BEGIN_DECLS

/**
* \author Guy Dickinson <guy.dickinson@nyu.edu>
* Encrypt/decrypt the data in inputbuf, writing the result to outputbuf.
* RC4 is symmetric (because it's a stream cipher), so we use the same
* function for both encryption and decryption.
*/
void rc4_cipher(rc4_state_t *state, const u_char *inputbuf, u_char *outputbuf, int buflength);

__END_DECLS