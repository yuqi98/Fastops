#include "Fastops_cpu.h"

using namespace std;
using namespace mkldnn;
using namespace Fastops;

memory::dim Data_cpu::product(const memory::dims &dims) {
return std::accumulate(dims.begin(), dims.end(), (memory::dim)1,
        std::multiplies<memory::dim>());
}

void Data_cpu::init(int argc, char **argv)
{
    engine eng_new(engine::kind::cpu, 0);
    eng = eng_new;
    stream s_new(eng);
    s = s_new;

}

void Data_cpu::set_data(std::vector<float> input, int batch, int input_channel, int width, int height)
{
    using tag = memory::format_tag;
    using dt = memory::data_type;
    memory::dims conv_src_tz = {batch, input_channel, width, height};
    memory user_src_memory = memory(
            { { conv_src_tz }, dt::f32, tag::nchw }, eng, input.data()); //offset_nchw(n, c, h, w) = n * CHW + c * HW + h * W + w
    
    current = user_src_memory;
    
    //dimensions also saved in 
    //memory_desc_t dst = get_desc(current) --> dst.dims = {batch, input_channel, height, width}

    dimensions.push_back(batch);
    dimensions.push_back(input_channel);
    dimensions.push_back(height);
    dimensions.push_back(width);
}

std::vector<float> Data_cpu::get_data(memory current_data)
{
    float *result = static_cast<float *>(current_data.get_data_handle());
    std::vector<float> re {result, result + dimensions[0]*dimensions[1]*dimensions[2]*dimensions[3]};
    return re;
}


void Data_cpu::convolution_layer(int batch, int input_channel, int width, int height, int output_channel, int kernel, int stride, int padding)
{
	using tag = memory::format_tag;
    using dt = memory::data_type;


    memory::dims conv_src_tz = { batch, input_channel, width, height };
    memory::dims conv_weights_tz = { output_channel, input_channel, kernel, kernel };
    memory::dims conv_bias_tz = { output_channel };
    memory::dims conv_dst_tz = { batch, output_channel, width, height};
    memory::dims conv_strides = { stride, stride };
    memory::dims conv_padding = { padding, padding };

    
    std::vector<float> conv_weights(product(conv_weights_tz));
    std::vector<float> conv_bias(product(conv_bias_tz));

    
    auto user_weights_memory
            = memory({ { conv_weights_tz }, dt::f32, tag::oihw }, eng,
                    conv_weights.data());
    auto conv_user_bias_memory = memory(
            { { conv_bias_tz }, dt::f32, tag::x }, eng, conv_bias.data());

    auto conv_src_md = memory::desc({ conv_src_tz }, dt::f32, tag::any);
    auto conv_bias_md = memory::desc({ conv_bias_tz }, dt::f32, tag::any);
    auto conv_weights_md
            = memory::desc({ conv_weights_tz }, dt::f32, tag::any);
    auto conv_dst_md = memory::desc({ conv_dst_tz }, dt::f32, tag::any);

    auto conv_desc = convolution_forward::desc(prop_kind::forward_inference,
            algorithm::convolution_direct, conv_src_md, conv_weights_md, conv_bias_md,
            conv_dst_md, conv_strides, conv_padding, conv_padding);

    auto conv_prim_desc = convolution_forward::primitive_desc(conv_desc, eng);

    auto conv_src_memory = current;
    if (conv_prim_desc.src_desc() != current.get_desc()) {
        conv_src_memory = memory(conv_prim_desc.src_desc(), eng);
        net.push_back(reorder(current, conv_src_memory));
        net_args.push_back({ { MKLDNN_ARG_FROM, current },
                { MKLDNN_ARG_TO, conv_src_memory } });
    }

    auto conv_weights_memory = user_weights_memory;
    if (conv_prim_desc.weights_desc() != user_weights_memory.get_desc()) {
        conv_weights_memory = memory(conv_prim_desc.weights_desc(), eng);
        reorder(user_weights_memory, conv_weights_memory)
                .execute(s, user_weights_memory, conv_weights_memory);
    }

    current = memory(conv_prim_desc.dst_desc(), eng);
	
	net.push_back(convolution_forward(conv_prim_desc));
    net_args.push_back({ { MKLDNN_ARG_SRC, conv_src_memory },
            { MKLDNN_ARG_WEIGHTS, conv_weights_memory },
            { MKLDNN_ARG_BIAS, conv_user_bias_memory },
            { MKLDNN_ARG_DST, current} });

}

void Data_cpu::relu(int batch, int input_channel, int width, int height, int output_channel)
{
    const float negative_slope = 1.0f;

    auto relu_desc = eltwise_forward::desc(prop_kind::forward_inference,
            algorithm::eltwise_relu, current.get_desc(),
            negative_slope);
    auto relu_prim_desc = eltwise_forward::primitive_desc(relu_desc, eng);

    net.push_back(eltwise_forward(relu_prim_desc));
    net_args.push_back({ { MKLDNN_ARG_SRC, current },
            { MKLDNN_ARG_DST, current } });
}



