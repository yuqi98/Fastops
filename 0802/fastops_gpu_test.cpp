
#include <cudnn.h>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <cassert>
#include <chrono>
#include "Fastops.h"

using namespace std;
using namespace Fastops;
using namespace chrono;


int main(int argc, char **argv)
{


	int n = 32;
  int layer = 10;


  Fastops_gpu mynet;

  
  int batch = 1;

  mynet.init_with_input(argc, argv, batch, 3, n, n);

  auto t1 = high_resolution_clock::now();

  mynet.convolution_layer_relu(batch, 3, n, n, 64, 5, 1, 2);

  for(int i = 1; i < 5; i++)
    mynet.convolutional_layer(cudnn, 1, 64, n, n, 64, 5, 1, 2);

  auto t2 = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(t2 - t1); 

  cout<<duration.count()<<endl;


	return 0;
}
