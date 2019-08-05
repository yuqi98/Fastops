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

class Fastops_cpu{
	public:
		
		engine eng;

		stream s;

		vector<primitive> net;

		vector<unordered_map<int, memory>> net_args;

		memory current;
		
		memory::dim product(const memory::dims &dims);

		void init_with_input(int argc, char **argv, int batch, int input_channel, int width, int height);

		void convolution_layer_relu(int batch, int input_channel, int width, int height, int output_channel, int kernel, int stride, int padding);


};

}

#endif