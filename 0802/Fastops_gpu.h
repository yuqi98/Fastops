#ifndef FASTOPS_GPU_H
#define FASTOPS_GPU_H


#include <assert.h>

#include <chrono>
#include <iostream>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>
#include <cmath>

#include <cudnn.h>

using namespace std;

namespace Fastops{

class Tensor_gpu{
	public:	

		float* init_kernel(int input_channel, int output_channel, int kernel);

		void set_data(std::vector<float> input, std::vector<int> input_size);

		std::vector<float> get_data();

		void convolution_layer(int output_channel, int kernel, int stride, int padding);

		void relu();

		float* get_value();

		std::vector<int> get_dim();
		
		void element_product(Tensor_gpu mul);

		void element_add(Tensor_gpu added);

	private:
		cudnnHandle_t cudnn;

		float* current;

		vector<int> dimensions;
};
	
}

#endif
