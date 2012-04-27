#!/bin/bash
rm -rf serial.txt
rm -rf parallel_noninterleaved*.txt
rm -rf parallel_interleaved*.txt
# number of iterations PER SIZE/TYPE
COUNTERMAIN=3
# initial buffer size, squared with each completed test cycle
SIZEMAIN=256

echo "baseline test"
COUNTER=$COUNTERMAIN
until [  $COUNTER -lt 1 ]; do
    /usr/bin/time --append --output=serial.txt cat cudatest.in | ./serial_test_streaming foo >> /dev/null
    let COUNTER-=1
done

echo "parallel, primitive  $SIZE"
COUNTER=$COUNTERMAIN
SIZE=$SIZEMAIN
until [  $SIZE = 16777215  ]; do
until [  $COUNTER -lt 1 ]; do
    /usr/bin/time --append --output=parallel_noninterleaved_$SIZE.txt cat cudatest.in | ./parallel_test_streaming foo $SIZE >> /dev/null
    let COUNTER-=1
done
let SIZE=$(echo "$SIZE^2" | bc)
if [ $SIZE -gt 16777214 ]; then
let SIZE=16777215
fi
done
/usr/bin/time --append --output=parallel_noninterleaved_$SIZE.txt cat cudatest.in | ./parallel_test_streaming foo $SIZE >> /dev/null

echo "parallel, interleaved $SIZE"
SIZE=$SIZEMAIN
until [  $SIZE = 16777215  ]; do
COUNTER=$COUNTERMAIN
until [  $COUNTER -lt 1 ]; do
    /usr/bin/time --append --output=parallel_interleaved_$SIZE.txt cat cudatest.in | ./parallel_test_streaming_interleaved foo $SIZE >> /dev/null
    let COUNTER-=1
done
let SIZE=$(echo "$SIZE^2" | bc)
if [ $SIZE -gt 16777214 ]; then
let SIZE=16777215
fi
done
/usr/bin/time --append --output=parallel_interleaved_$SIZE.txt cat cudatest.in | ./parallel_test_streaming_interleaved foo $SIZE >> /dev/null
