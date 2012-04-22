# 
# Copyright 2012 Guy Dickinson <guy.dickinson@nyu.edu>,
# William Ward <wwward@nyu.edu> 
# 
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
# 
#        http://www.apache.org/licenses/LICENSE-2.0
# 
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# Makefile for CUDA GPU Project
# Many thanks to T. Hudson for the help creating this

CC		= nvcc
CFLAGS		= \
	-Xcompiler '-Wall -Wextra' \
	-I../include \

LDFLAGS		= $(CFLAGS)

OBJS 		= \
	rc4_parallel.o \
	rc4_common.o \
	rc4_serial.o

all: parallel_test serial_test serial_test_streaming

parallel_test: $(OBJS)
	$(CC) -o $@ $(LDFLAGS) $^ rc4_parallel_test.cu

parallel_test_streaming: $(OBJS)
	$(CC) -o $@ $(LDFLAGS) $^ rc4_parallel_streaming.cu

serial_test: $(OBJS)
	$(CC) -o $@ $(LDFLAGS) $^ rc4_serial_test.c

serial_test_streaming: $(OBJS)
	$(CC) -o $@ $(LDFLAGS) $^ rc4_serial_streaming.c

%.o: %.cu
	$(CC) $(CFLAGS) -c -o $@ $<
%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f *.o .*.d

-include .*.d
