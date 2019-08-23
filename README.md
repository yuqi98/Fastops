# Fastops

library compile with mkldnn

g++ -std=c++11 -fpic -c -I path_to_mkldnn/include/ Fastops_cpu.cpp -lmkldnn -ltbb
g++ -shared -o libFastops_cpu.so Fastops_cpu.o -I path_to_mkldnn/include/ -lmkldnn -ltbb

library compile with cudnn:

g++ -c Fastops_gpu.cpp -I /usr/local/cuda/include -L /usr/local/cuda/lib64 -L /usr/local/lib
ar rvs Fastops_gpu.a Fastops_gpu.o

compile with actually code

cpu example:
g++ -std=c++11 fastops_test_cpu.cpp -o fastops_test_cpu -I path_to_mkldnn/include/ -lmkldnn  -L ./ -lFastops_cpu -ltbb

gpu example:

/usr/local/cuda/bin/nvcc -arch=sm_35 -std=c++11 -I /usr/local/cuda/include -L /usr/local/cuda/lib64 -L /usr/local/lib -L /usr/local/cuda/lib64/libcudart_static.a -L ./ fastops_gpu_test.cpp -lcudnn Fastops_gpu.a



