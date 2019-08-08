#include "Fastops_gpu.h"
using namespace std;
using namespace Fastops;
using namespace chrono;


#define checkCUDNN(expression)                               \
{                                                          \
  cudnnStatus_t status = (expression);                     \
  if (status != CUDNN_STATUS_SUCCESS) {                    \
    std::cerr << "Error on line " << __LINE__ << ": "      \
              << cudnnGetErrorString(status) << std::endl; \
    std::exit(EXIT_FAILURE);                               \
  }                                                        \
}


float* Data_gpu::init_kernel(int input_channel, int output_channel, int kernel)
{
  srand(time(NULL));
  float min = -1;
  float max = 1;

  float h_kernel[input_channel*output_channel*kernel*kernel];
  for (int kernels = 0; kernels < input_channel; ++kernels) {
    for (int channel = 0; channel < output_channel; ++channel) {
      for (int row = 0; row < kernel; ++row) {
        for (int column = 0; column < kernel; ++column) {
          h_kernel[kernel+channel*input_channel+row*input_channel*output_channel+column*input_channel*output_channel*kernel] = min + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(max-min)));
        }
      }
    }
  }
  return h_kernel;
}

void Data_gpu::init(int argc, char **argv)
{
  int gpu_id = (argc > 2) ? std::atoi(argv[2]) : 0;
  cudaSetDevice(gpu_id);
  cudnnHandle_t cudnn_new{nullptr}; 
  cudnnCreate(&cudnn_new);

  cudnn = cudnn_new;
}

void Data_gpu::set_data(vector<float> input, int batch, int input_channel, int width, int height)
{  
  current = &input[0];
  dimensions.push_back(batch);
  dimensions.push_back(input_channel);
  dimensions.push_back(height);
  dimensions.push_back(width);
}

std::vector<float> Data_gpu::get_data(float* current_data)
{
  std::vector<float> result {current_data, current_data + dimensions[0]*dimensions[1]*dimensions[2]*dimensions[3]};
  return result;
}

void Data_gpu::convolution_layer(int batch, int input_channel, int width, int height, int output_channel, int kernel, int stride, int padding)
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

    float* h_kernel = init_kernel(input_channel, output_channel, kernel);

		float* d_kernel{nullptr};
		cudaMalloc(&d_kernel, sizeof(h_kernel));
		cudaMemcpy(d_kernel, h_kernel, sizeof(h_kernel), cudaMemcpyHostToDevice);

		float* d_input{nullptr};
  	cudaMalloc(&d_input, input_image_bytes);
  	cudaMemcpy(d_input, current, input_image_bytes, cudaMemcpyHostToDevice);

  	float* d_output{nullptr};
  	cudaMalloc(&d_output, output_image_bytes);
  	cudaMemset(d_output, 0, output_image_bytes);
    
    const float alpha = 1.0f, beta = 0.0f;

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
	current = h_output;

}

void Data_gpu::relu(int batch, int input_channel, int width, int height, int output_channel)
{

  int input_image_bytes = batch * input_channel * height * width * sizeof(float);
  int output_image_bytes = batch * output_channel * height * width * sizeof(float);

  float* d_input{nullptr};
  cudaMalloc(&d_input, input_image_bytes);
  cudaMemcpy(d_input, current, input_image_bytes, cudaMemcpyHostToDevice);
  
  float* d_output{nullptr};
  cudaMalloc(&d_output, output_image_bytes);
  cudaMemset(d_output, 0, output_image_bytes);

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

  const float alpha = 1.0f, beta = 0.0f;
  cudnnActivationDescriptor_t activation_descriptor;
  checkCUDNN(cudnnCreateActivationDescriptor(&activation_descriptor));
  checkCUDNN(cudnnSetActivationDescriptor(activation_descriptor,
                             /*mode=*/CUDNN_ACTIVATION_RELU,
                       /*reluNanOpt=*/CUDNN_PROPAGATE_NAN,
                        /*relu_coef=*/0));

  checkCUDNN(cudnnActivationForward(cudnn,
                                activation_descriptor,
                                &alpha,
                                input_descriptor,
                                d_input,
                                &beta,
                                output_descriptor,
                                d_output));
  float* h_output = new float[output_image_bytes];

  cudaMemcpy(h_output, d_output, output_image_bytes, cudaMemcpyDeviceToHost);
  
  cudaFree(d_input);
  cudaFree(d_output);

  cudnnDestroyTensorDescriptor(input_descriptor);
  cudnnDestroyTensorDescriptor(output_descriptor);
  checkCUDNN(cudnnDestroyActivationDescriptor(activation_descriptor));

  current = h_output;
}