#ifndef FASTOPS_H
#define FASTOPS_H


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


//int multi(int n);
memory::dim product(const memory::dims &dims);

memory cpu_input_layer(engine eng, const memory::dim batch, int input_channel, int output_channel, int width, int height);

memory cpu_convolution_layer_relu(vector<primitive> &net, vector<unordered_map<int, memory>> &net_args, engine eng, const memory::dim batch, int input_channel, int width, int height, int output_channel, int kernel, int stride, int padding, memory conv_dst_memory);



}

#endif