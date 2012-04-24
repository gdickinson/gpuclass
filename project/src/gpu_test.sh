#!/bin/bash
rm -rf ~/serial.txt
rm -rf ~/parallel_noninterleaved.txt
rm -rf ~/parallel_interleaved.txt

COUNTER=10
until [  $COUNTER -lt 1 ]; do
    /usr/bin/time --append --output=~/serial.txt cat ~/cudatest.in | serial_test_streaming foo >> /dev/null
    let COUNTER-=1
done
COUNTER=10
until [  $COUNTER -lt 1 ]; do
    /usr/bin/time --append --output=~/parallel_noninterleaved.txt cat ~/cudatest.in | parallel_test_streaming foo >> /dev/null
    let COUNTER-=1
done
COUNTER=10
until [  $COUNTER -lt 1 ]; do
    /usr/bin/time --append --output=~/parallel_interleaved.txt cat ~/cudatest.in | parallel_test_streaming_interleaved foo 512 >> /dev/null
    let COUNTER-=1
done
