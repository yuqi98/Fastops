#ifndef FASTOPS_H
#define FASTOPS_H


#include <assert.h>

#include <chrono>
#include <iostream>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

//#include "mkldnn.hpp"
#include <cudnn.h>


using namespace std;
//using namespace mkldnn;

namespace Fastops{

//cpu part:

	memory::dim product(const memory::dims &dims);

	memory cpu_input_layer(engine eng, const memory::dim batch, int input_channel, int output_channel, int width, int height);

	memory cpu_convolution_layer_relu(vector<primitive> &net, vector<unordered_map<int, memory>> &net_args, engine eng, const memory::dim batch, int input_channel, int width, int height, int output_channel, int kernel, int stride, int padding, memory conv_dst_memory);

//gpu part:
	cudnnHandle_t gpu_init(int argc, char **argv);

	float* gpu_convolutional_layer(cudnnHandle_t cudnn, int batch, int input_channel, int width, int height, int output_channel, int kernel, int stride, int padding, float* h_kernel, float* input_image);

}

#endif