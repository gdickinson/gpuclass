#!/bin/bash
rm -rf serial.txt
rm -rf parallel_noninterleaved.txt
rm -rf parallel_interleaved.txt
COUNTERMAIN=10

echo "baseline test"
COUNTER=$COUNTERMAIN
until [  $COUNTER -lt 1 ]; do
    /usr/bin/time --append --output=serial.txt cat cudatest.in | ./serial_test_streaming foo >> /dev/null
    let COUNTER-=1
done

echo "parallel, primitive 512"
COUNTER=$COUNTERMAIN
until [  $COUNTER -lt 1 ]; do
    /usr/bin/time --append --output=parallel_noninterleaved_512.txt cat cudatest.in | ./parallel_test_streaming foo 512 >> /dev/null
    let COUNTER-=1
done

echo "parallel, interleaved 512"
COUNTER=$COUNTERMAIN
until [  $COUNTER -lt 1 ]; do
    /usr/bin/time --append --output=parallel_interleaved_512.txt cat cudatest.in | ./parallel_test_streaming_interleaved foo 512 >> /dev/null
    let COUNTER-=1
done

echo "parallel, primitive  max 16,777,215"
COUNTER=$COUNTERMAIN
until [  $COUNTER -lt 1 ]; do
    /usr/bin/time --append --output=parallel_noninterleaved_16.7.txt cat cudatest.in | ./parallel_test_streaming foo 16777215 >> /dev/null
    let COUNTER-=1
done

echo "parallel, interleaved max 16,777,215" 
COUNTER=$COUNTERMAIN
until [  $COUNTER -lt 1 ]; do
    /usr/bin/time --append --output=parallel_interleaved_16.7.txt cat cudatest.in | ./parallel_test_streaming_interleaved foo 16777215 >> /dev/null
    let COUNTER-=1
done
