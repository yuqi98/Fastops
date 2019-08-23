#ifndef FASTOPS_CPU_H
#define FASTOPS_CPU_H

#include <assert.h>

#include <chrono>
#include <iostream>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#include "mkldnn.hpp"
#include <tbb/parallel_for.h>

using namespace std;
using namespace mkldnn;

namespace Fastops{

class Tensor_cpu{
	public:
		engine eng;
		stream s;

		void set_data(std::vector<float> input, std::vector<int> input_size, int flag);

		std::vector<float> get_data();

		void convolution_layer(vector<primitive> &net, vector<unordered_map<int, memory>> &net_args, Tensor_cpu kernel, Tensor_cpu bias, int stride, int padding);

		void relu(vector<primitive> &net, vector<unordered_map<int, memory>> &net_args);

		memory get_value();

		std::vector<int> get_dim();
		
		void element_product(Tensor_cpu mul);

		void element_add(Tensor_cpu added);

	private:
		memory current;
		
		vector<int> dimensions;
};

}

#endif
