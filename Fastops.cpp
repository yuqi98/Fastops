#include "Fastops.h"

using namespace std;
using namespace mkldnn;

#define checkCUDNN(expression)                               \
{                                                          \
	cudnnStatus_t status = (expression);                     \
	if (status != CUDNN_STATUS_SUCCESS) {                    \
	  std::cerr << "Error on line " << __LINE__ << ": "      \
	            << cudnnGetErrorString(status) << std::endl; \
	  std::exit(EXIT_FAILURE);                               \
	}                                                        \
}


namespace Fastops{
	
	//cpu part using mkl-dnn
	

	memory::dim product(const memory::dims &dims) {
    return std::accumulate(dims.begin(), dims.end(), (memory::dim)1,
            std::multiplies<memory::dim>());
	}

	memory cpu_input_layer(engine eng, const memory::dim batch, int input_channel, int output_channel, int width, int height)
	{
		using tag = memory::format_tag;
	    using dt = memory::data_type;
		std::vector<float> user_src(batch * input_channel * width * height);
	    memory::dims conv_src_tz = {batch, input_channel, width, height};
	    memory user_src_memory = memory(
	            { { conv_src_tz }, dt::f32, tag::nchw }, eng, user_src.data());
	    return user_src_memory;

	}
	memory cpu_convolution_layer_relu(vector<primitive> &net, vector<unordered_map<int, memory>> &net_args, engine eng, const memory::dim batch, int input_channel, int width, int height, int output_channel, int kernel, int stride, int padding, memory conv_dst_memory)
	{
		using tag = memory::format_tag;
	    using dt = memory::data_type;
	    stream s(eng);


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

	    auto conv_src_memory = conv_dst_memory;
	    if (conv_prim_desc.src_desc() != conv_dst_memory.get_desc()) {
	        conv_src_memory = memory(conv_prim_desc.src_desc(), eng);
	        net.push_back(reorder(conv_dst_memory, conv_src_memory));
	        net_args.push_back({ { MKLDNN_ARG_FROM, conv_dst_memory },
	                { MKLDNN_ARG_TO, conv_src_memory } });
	    }

	    auto conv_weights_memory = user_weights_memory;
	    if (conv_prim_desc.weights_desc() != user_weights_memory.get_desc()) {
	        conv_weights_memory = memory(conv_prim_desc.weights_desc(), eng);
	        reorder(user_weights_memory, conv_weights_memory)
	                .execute(s, user_weights_memory, conv_weights_memory);
	    }

	    conv_dst_memory = memory(conv_prim_desc.dst_desc(), eng);
		
		net.push_back(convolution_forward(conv_prim_desc));
        net_args.push_back({ { MKLDNN_ARG_SRC, conv_src_memory },
                { MKLDNN_ARG_WEIGHTS, conv_weights_memory },
                { MKLDNN_ARG_BIAS, conv_user_bias_memory },
                { MKLDNN_ARG_DST, conv_dst_memory} });


        const float negative_slope = 1.0f;

        auto relu_desc = eltwise_forward::desc(prop_kind::forward_inference,
                algorithm::eltwise_relu, conv_dst_memory.get_desc(),
                negative_slope);
        auto relu_prim_desc = eltwise_forward::primitive_desc(relu_desc, eng);

        net.push_back(eltwise_forward(relu_prim_desc));
        net_args.push_back({ { MKLDNN_ARG_SRC, conv_dst_memory },
                { MKLDNN_ARG_DST, conv_dst_memory } });

		return conv_dst_memory;
	}
	

	//gpu part:

	cudnnHandle_t gpu_init(int argc, char **argv)
	{
		int gpu_id = (argc > 2) ? std::atoi(argv[2]) : 0;
		cudaSetDevice(gpu_id);
		cudnnHandle_t cudnn{nullptr};	
		cudnnCreate(&cudnn);

		return cudnn;
	}

	float* gpu_convolutional_layer(cudnnHandle_t cudnn, int batch, int input_channel, int width, int height, int output_channel, int kernel, int stride, int padding, float* h_kernel, float* input_image)
	{
		cudnnTensorDescriptor_t input_descriptor;
		checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
		checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                            /*format=*/CUDNN_TENSOR_NHWC,
                          /*dataType=*/CUDNN_DATA_FLOAT,
                        /*batch_size=*/batch,
                          /*channels=*/input_channel,
                      /*image_height=*/height,
                       /*image_width=*/width));

  		cudnnTensorDescriptor_t output_descriptor;
  		checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
  		checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                            /*format=*/CUDNN_TENSOR_NHWC,
                          /*dataType=*/CUDNN_DATA_FLOAT,
                        /*batch_size=*/batch,
                          /*channels=*/output_channel,
                      /*image_height=*/height,
                       /*image_width=*/width));

		cudnnFilterDescriptor_t kernel_descriptor;
	    checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
	    checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
	                          /*dataType=*/CUDNN_DATA_FLOAT,
	                            /*format=*/CUDNN_TENSOR_NCHW,
	                      /*out_channels=*/output_channel,
	                       /*in_channels=*/input_channel,
	                     /*kernel_height=*/kernel,
	                      /*kernel_width=*/kernel));

		cudnnConvolutionDescriptor_t convolution_descriptor;
	    checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
	    checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
	                              /*pad_height=*/padding,
	                               /*pad_width=*/padding,
	                         /*vertical_stride=*/stride,
	                       /*horizontal_stride=*/stride,
	                         /*dilation_height=*/stride,
	                          /*dilation_width=*/stride,
	                                    /*mode=*/CUDNN_CROSS_CORRELATION,
	                             /*computeType=*/CUDNN_DATA_FLOAT));
	    
	    cudnnConvolutionFwdAlgo_t convolution_algorithm;
  		checkCUDNN(
    	cudnnGetConvolutionForwardAlgorithm(cudnn,
                                        input_descriptor,
                                        kernel_descriptor,
                                        convolution_descriptor,
                                        output_descriptor,
                                        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                        /*memoryLimitInBytes=*/0,
                                        &convolution_algorithm));

  		const float alpha = 1.0f, beta = 0.0f;
  		cudnnActivationDescriptor_t activation_descriptor;
  		checkCUDNN(cudnnCreateActivationDescriptor(&activation_descriptor));
  		checkCUDNN(cudnnSetActivationDescriptor(activation_descriptor,
                                 /*mode=*/CUDNN_ACTIVATION_RELU,
                           /*reluNanOpt=*/CUDNN_PROPAGATE_NAN,
                            /*relu_coef=*/0));
  		
  		size_t workspace_bytes = 0;
  		checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                                   input_descriptor,
                                                   kernel_descriptor,
                                                   convolution_descriptor,
                                                   output_descriptor,
                                                   convolution_algorithm,
                                                   &workspace_bytes));
  		assert(workspace_bytes > 0);

  		void* d_workspace{nullptr};
  		cudaMalloc(&d_workspace, workspace_bytes);

  		int input_image_bytes = batch * input_channel * height * width * sizeof(float);
  		int output_image_bytes = batch * output_channel * height * width * sizeof(float);

  		float* d_kernel{nullptr};
  		cudaMalloc(&d_kernel, sizeof(h_kernel));
  		cudaMemcpy(d_kernel, h_kernel, sizeof(h_kernel), cudaMemcpyHostToDevice);

  		float* d_input{nullptr};
    	cudaMalloc(&d_input, input_image_bytes);
    	cudaMemcpy(d_input, input_image, input_image_bytes, cudaMemcpyHostToDevice);

    	float* d_output{nullptr};
    	cudaMalloc(&d_output, output_image_bytes);
    	cudaMemset(d_output, 0, output_image_bytes);

    	checkCUDNN(cudnnConvolutionForward(cudnn,
                                     &alpha,
                                     input_descriptor,
                                     d_input,
                                     kernel_descriptor,
                                     d_kernel,
                                     convolution_descriptor,
                                     convolution_algorithm,
                                     d_workspace,
                                     workspace_bytes,
                                     &beta,
                                     output_descriptor,
                                     d_output));
    	checkCUDNN(cudnnActivationForward(cudnn,
                                      activation_descriptor,
                                      &alpha,
                                      output_descriptor,
                                      d_output,
                                      &beta,
                                      output_descriptor,
                                      d_output));
    	
    	float* h_output = new float[output_image_bytes];

   		cudaMemcpy(h_output, d_output, output_image_bytes, cudaMemcpyDeviceToHost);
    	
    	cudaFree(d_input);
    	cudaFree(d_output);
    	cudaFree(d_kernel);
  		cudaFree(d_workspace);

		cudnnDestroyTensorDescriptor(input_descriptor);
		cudnnDestroyTensorDescriptor(output_descriptor);
		cudnnDestroyFilterDescriptor(kernel_descriptor);
		cudnnDestroyConvolutionDescriptor(convolution_descriptor);
		checkCUDNN(cudnnDestroyActivationDescriptor(activation_descriptor));

		return(h_output);

	}


}