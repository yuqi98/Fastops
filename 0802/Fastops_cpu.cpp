#include "Fastops_cpu.h"

using namespace std;
using namespace mkldnn;
using namespace Fastops;


void Tensor_cpu::set_data(std::vector<float> input, std::vector<int> input_size, int flag)
{
    engine new_eng(engine::kind::cpu, 0);
    this->eng = new_eng;
    stream new_stream(this->eng);
    this->s = new_stream;

    using tag = memory::format_tag;
    using dt = memory::data_type;

    this->dimensions.push_back(input_size[0]);

    for(int i = 1; i < input_size.size(); i++)
    {
        this->dimensions.push_back(input_size[i]);
    }
    memory::dims conv_src_tz;
    if(input_size.size() == 4)
        conv_src_tz = {this->dimensions[0], this->dimensions[1], this->dimensions[2], this->dimensions[3]};
    if(input_size.size() == 3)
        conv_src_tz = {this->dimensions[0], this->dimensions[1], this->dimensions[2]};
    if(input_size.size() == 2)
        conv_src_tz = {this->dimensions[0], this->dimensions[1]};
    if(input_size.size() == 1)
        conv_src_tz = {this->dimensions[0]};

    memory user_src_memory;
    if(flag == 1)
        user_src_memory = memory(
                { { conv_src_tz }, dt::f32, tag::nchw }, this->eng, input.data()); //offset_nchw(n, c, h, w) = n * CHW + c * HW + h * W + w
    if(flag == 2)
        user_src_memory = memory(
                { { conv_src_tz }, dt::f32, tag::oihw }, this->eng, input.data());   
    if(flag == 3)
        user_src_memory = memory(
                { { conv_src_tz }, dt::f32, tag::x }, this->eng, input.data()); 

    this->current = user_src_memory;
    
    //dimensions also saved in 
    //memory_desc_t dst = get_desc(current) --> dst.dims = {batch, input_channel, height, width}

}

std::vector<float> Tensor_cpu::get_data()
{
    float *result = static_cast<float *>(this->current.get_data_handle());
    int total_size = 1;
    for(int i = 0; i < this->dimensions.size(); i++)
    {
        total_size *= this->dimensions[i];
    }
    std::vector<float> re {result, result + total_size};
    return re;
}


void Tensor_cpu::convolution_layer(vector<primitive> &net, vector<unordered_map<int, memory>> &net_args, Tensor_cpu kernel, Tensor_cpu bias, int stride, int padding)
{
	using tag = memory::format_tag;
    using dt = memory::data_type;


    memory::dims conv_src_tz = { this->dimensions[0], this->dimensions[1], this->dimensions[2], this->dimensions[3]};
    std::vector<int> dims_kernel = kernel.get_dim();
    std::vector<int> dims_bias = bias.get_dim();
    memory::dims conv_weights_tz = {dims_kernel[0], dims_kernel[1], dims_kernel[2], dims_kernel[3]};
    memory::dims conv_bias_tz = {dims_bias[0]};
    memory::dims conv_dst_tz = { this->dimensions[0], dims_kernel[0], this->dimensions[2], this->dimensions[3]};
    memory::dims conv_strides = { stride, stride };
    memory::dims conv_padding = { padding, padding };

    
    std::vector<float> conv_weights = kernel.get_data();
    std::vector<float> conv_bias = bias.get_data();
    
    auto user_weights_memory
            = memory({ { conv_weights_tz }, dt::f32, tag::oihw }, this->eng,
                    conv_weights.data());
    auto conv_user_bias_memory = memory(
            { { conv_bias_tz }, dt::f32, tag::x }, this->eng, conv_bias.data());

    auto conv_src_md = memory::desc({ conv_src_tz }, dt::f32, tag::any);
    auto conv_bias_md = memory::desc({ conv_bias_tz }, dt::f32, tag::any);
    auto conv_weights_md
            = memory::desc({ conv_weights_tz }, dt::f32, tag::any);
    auto conv_dst_md = memory::desc({ conv_dst_tz }, dt::f32, tag::any);

    auto conv_desc = convolution_forward::desc(prop_kind::forward_inference,
            algorithm::convolution_direct, conv_src_md, conv_weights_md, conv_bias_md,
            conv_dst_md, conv_strides, conv_padding, conv_padding);

    auto conv_prim_desc = convolution_forward::primitive_desc(conv_desc, this->eng);

    auto conv_src_memory = this->current;
    if (conv_prim_desc.src_desc() != this->current.get_desc()) {
        conv_src_memory = memory(conv_prim_desc.src_desc(), this->eng);
        net.push_back(reorder(this->current, conv_src_memory));
        net_args.push_back({ { MKLDNN_ARG_FROM, this->current },
                { MKLDNN_ARG_TO, conv_src_memory } });
    }

    auto conv_weights_memory = user_weights_memory;
    if (conv_prim_desc.weights_desc() != user_weights_memory.get_desc()) {
        conv_weights_memory = memory(conv_prim_desc.weights_desc(), this->eng);
        reorder(user_weights_memory, conv_weights_memory)
                .execute(this->s, user_weights_memory, conv_weights_memory);
    }

    this->current = memory(conv_prim_desc.dst_desc(), this->eng);
	this->dimensions[1] = dims_kernel[0];

	net.push_back(convolution_forward(conv_prim_desc));
    net_args.push_back({ { MKLDNN_ARG_SRC, conv_src_memory },
            { MKLDNN_ARG_WEIGHTS, conv_weights_memory },
            { MKLDNN_ARG_BIAS, conv_user_bias_memory },
            { MKLDNN_ARG_DST, this->current} });

}

void Tensor_cpu::relu(vector<primitive> &net, vector<unordered_map<int, memory>> &net_args)
{
    const float negative_slope = 1.0f;

    auto relu_desc = eltwise_forward::desc(prop_kind::forward_inference,
            algorithm::eltwise_relu, this->current.get_desc(),
            negative_slope);
    auto relu_prim_desc = eltwise_forward::primitive_desc(relu_desc, this->eng);

    net.push_back(eltwise_forward(relu_prim_desc));
    net_args.push_back({ { MKLDNN_ARG_SRC, this->current },
            { MKLDNN_ARG_DST, this->current } });
}

memory Tensor_cpu::get_value()
{
    return this->current;
}

std::vector<int> Tensor_cpu::get_dim()
{
    return this->dimensions;   
}

void Tensor_cpu::element_product(Tensor_cpu mul)
{
    memory value1 = this->get_value();
    memory value2 = mul.get_value();
    std::vector<int> dims1 = this->get_dim();
    std::vector<int> dims2 = mul.get_dim();

    float* a = static_cast<float *>(value1.get_data_handle());
    float* b = static_cast<float *>(value2.get_data_handle());

    int size = dims1[0]*dims1[1]*dims1[2]*dims1[3];

    std::vector<float> re {a, a + size};
    std::vector<float> re2 {b, b + size};

    tbb::parallel_for( tbb::blocked_range<int>(0,re.size()),
                       [&](tbb::blocked_range<int> r)
    {
        for (int i=r.begin(); i<r.end(); ++i)
        {
            re[i] = re[i]*re2[i];
        }
    });

    using tag = memory::format_tag;
    using dt = memory::data_type;
    memory::dims conv_src_tz = {dims1[0], dims1[1], dims1[2], dims1[3]};
    memory user_src_memory = memory(
            { { conv_src_tz }, dt::f32, tag::nchw }, this->eng, re.data()); //offset_nchw(n, c, h, w) = n * CHW + c * HW + h * W + w
    
    this->current = user_src_memory;
    
}

void Tensor_cpu::element_add(Tensor_cpu added)
{
    memory value1 = this->get_value();
    memory value2 = added.get_value();
    std::vector<int> dims1 = this->get_dim();
    std::vector<int> dims2 = added.get_dim();

    float* a = static_cast<float *>(value1.get_data_handle());
    float* b = static_cast<float *>(value2.get_data_handle());

    int size = dims1[0]*dims1[1]*dims1[2]*dims1[3];

    std::vector<float> re {a, a + size};
    std::vector<float> re2 {b, b + size};

    tbb::parallel_for( tbb::blocked_range<int>(0,re.size()),
                       [&](tbb::blocked_range<int> r)
    {
        for (int i=r.begin(); i<r.end(); ++i)
        {
            re[i] = re[i]+re2[i];
        }
    });

    using tag = memory::format_tag;
    using dt = memory::data_type;
    memory::dims conv_src_tz = {dims1[0], dims1[1], dims1[2], dims1[3]};
    memory user_src_memory = memory(
            { { conv_src_tz }, dt::f32, tag::nchw }, this->eng, re.data()); //offset_nchw(n, c, h, w) = n * CHW + c * HW + h * W + w
    
    this->current = user_src_memory;
    
}




