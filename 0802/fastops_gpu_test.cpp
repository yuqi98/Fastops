
#include <cudnn.h>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <cassert>
#include <chrono>
#include "Fastops_gpu.h"

using namespace std;
using namespace Fastops;
using namespace chrono;


int main(int argc, char **argv)
{

  int gpu_id = (argc > 2) ? std::atoi(argv[2]) : 0;
  cudaSetDevice(gpu_id);
  int n = 32;
  int layer = 10;
  int batch = 1;

  
  Tensor_gpu mynet;

  srand(time(NULL));
  float min = -1;
  float max = 1;

  vector<float> input_image(batch*3*n*n);
  for(int i = 0; i < 3; i++)
    for(int j = 0; j < n; j++)
      for(int k = 0; k < n; k++)
        input_image[i+j*3+k*3*n] = min + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(max-min)));

  cout<<2<<endl;
  vector<int> input_size;
  input_size.push_back(batch);
  input_size.push_back(3);
  input_size.push_back(n);
  input_size.push_back(n);
  mynet.set_data(input_image,input_size);
  cout<<3<<endl;
  auto t1 = high_resolution_clock::now();

  mynet.convolution_layer(64, 5, 1, 2);
  cout<<4<<endl;
  mynet.relu();
  cout<<5<<endl;
  for(int i = 1; i < 5; i++){
    mynet.convolution_layer(64, 5, 1, 2);
    mynet.relu();
  }

  auto t2 = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(t2 - t1); 

  vector<float> result = mynet.get_data();
  cout<<duration.count()<<endl;


	return 0;
}
