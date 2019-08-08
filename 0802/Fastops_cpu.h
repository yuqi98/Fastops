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

using namespace std;
using namespace mkldnn;

namespace Fastops{

class Data_cpu{
	public:
		
		engine eng;

		stream s;

		vector<primitive> net;

		vector<unordered_map<int, memory>> net_args;

		memory current;
		
		vector<int> dimensions;

		memory::dim product(const memory::dims &dims);

		void init(int argc, char **argv);

		void set_data(std::vector<float> input, int batch, int input_channel, int width, int height);

		std::vector<float> get_data(memory current_data);

		void convolution_layer(int batch, int input_channel, int width, int height, int output_channel, int kernel, int stride, int padding);

		void relu(int batch, int input_channel, int width, int height, int output_channel);

};

}

#endif