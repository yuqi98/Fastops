#ifndef FASTOPS_GPU_H
#define FASTOPS_GPU_H


#include <assert.h>

#include <chrono>
#include <iostream>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#include <cudnn.h>

using namespace std;

namespace Fastops{

class Fastops_gpu{
	public:	

		cudnnHandle_t cudnn;

		float* current;

		float* init_kernel(int input_channel, int output_channel, int kernel);

		void init_with_input(int argc, char **argv, int batch, int input_channel, int width, int height);		

		void convolutional_layer(int batch, int input_channel, int width, int height, int output_channel, int kernel, int stride, int padding);

};
	
}
