
#include <cudnn.h>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <cassert>
#include <chrono>
#include "Fastops.h"

using namespace std;
using namespace chrono;


int main(int argc, char **argv)
{


	int n = 32;
  int layer = 10;

  srand(time(NULL));
  float min = -1;
  float max = 1;

  cudnnHandle_t cudnn = Fastops::gpu_init(argc, argv);
  float h_kernel[64*3*5*5];
  for (int kernel = 0; kernel < 64; ++kernel) {
    for (int channel = 0; channel < 3; ++channel) {
      for (int row = 0; row < 5; ++row) {
        for (int column = 0; column < 5; ++column) {
          h_kernel[kernel+channel*64+row*3*64+column*64*3*5] = min + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(max-min)));
        }
      }
    }
  }
  float h_kernel2[64*64*5*5];
  for (int kernel = 0; kernel < 64; ++kernel) {
    for (int channel = 0; channel < 64; ++channel) {
      for (int row = 0; row < 5; ++row) {
        for (int column = 0; column < 5; ++column) {
          h_kernel2[kernel+channel*64+row*64*64+column*64*64*5] = min + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(max-min)));
        }
      }
    }
  }

  float input_image[3*n*n];
    for(int i = 0; i < 3; i++)
      for(int j = 0; j < n; j++)
        for(int k = 0; k < n; k++)
          input_image[i+j*3+k*3*n] = min + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(max-min)));

  auto t1 = high_resolution_clock::now();
  float* h_output = Fastops::gpu_convolutional_layer(cudnn, 1, 3, n, n, 64, 5, 1, 2, h_kernel, input_image);

  for(int i = 1; i < 5; i++)
    h_output = Fastops::gpu_convolutional_layer(cudnn, 1, 64, n, n, 64, 5, 1, 2, h_kernel2, h_output);

  auto t2 = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(t2 - t1); 

  cout<<duration.count()<<endl;


	return 0;
}
