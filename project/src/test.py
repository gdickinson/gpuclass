import time
import sys
import subprocess

bufsize = 512
infile = open("cudatest.in","r")
out = open("/dev/null","w")

processes = [ "./serial_test_streaming", "./parallel_test_streaming_interleaved", "./parallel_test_streaming" ]

for process in processes:
	

	while (bufsize < 16777215):
		start = time.time()
		subprocess.call([process, "key", str(bufsize)], stdin=infile, stdout=out)
		end = time.time()
		print ("%s,%d,%f" % (process,bufsize,(end-start)))
		infile.seek(0)
		bufsize = (bufsize * 2)	

	bufsize = 512
