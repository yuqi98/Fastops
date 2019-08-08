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

class Data_gpu{
	public:	

		cudnnHandle_t cudnn;

		float* current;

		vector<int> dimensions;

		float* init_kernel(int input_channel, int output_channel, int kernel);

		void init(int argc, char **argv);		

		void set_data(std::vector<float> input, int batch, int input_channel, int width, int height);

		std::vector<float> get_data(float* current_data);

		void convolution_layer(int batch, int input_channel, int width, int height, int output_channel, int kernel, int stride, int padding);

		void relu(int batch, int input_channel, int width, int height, int output_channel);

};
	
}

#endif
