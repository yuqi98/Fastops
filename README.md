# Fastops

library compile with mkldnn

g++ -std=c++11 -fpic -c -I path_to_mkldnn/include/ Fastops.cpp -lmkldnn
g++ -shared -o libFastops.so Fastops.o -I path_to_mkldnn/include/ -lmkldnn

library compile with cudnn:

g++ -std=c++11 -O2 -fpic -c -I /usr/local/cuda/include -L /usr/local/cuda/lib64 -L/usr/local/lib Fastops.cpp -lcudnn

g++ -shared -o libFastops.so Fastops.o -I /usr/local/cuda/include -L /usr/local/cuda/lib64 -L/usr/local/lib -lcudnn

compile with actually code

cpu example:
g++ -std=c++11 fastops_test_cpu.cpp -o fastops_test_cpu -I path_to_mkldnn/include/ -lmkldnn  -L ./ -lFastops

gpu example:

/usr/local/cuda/bin/nvcc -arch=sm_35 -std=c++11 -O2 -I /usr/local/cuda/include -L /usr/local/cuda/lib64 -L /usr/local/lib -L ./ fastops_gpu_test.cpp -lcudnn -lFastops



