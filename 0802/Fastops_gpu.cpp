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


float* Tensor_gpu::init_kernel(int input_channel, int output_channel, int kernel)
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

void Tensor_gpu::set_data(std::vector<float> input, std::vector<int> input_size)
{
  cudnnHandle_t cudnn_new{nullptr}; 
  cudnnCreate(&cudnn_new);
  this->cudnn = cudnn_new;  

  this->current = &input[0];

  for(int i = 0; i < input_size.size(); i++)
    this->dimensions.push_back(input_size[i]);
}

std::vector<float> Tensor_gpu::get_data()
{
  int total_size = this->dimensions[0];
  for(int i = 1; i < this->dimensions.size(); i++)
    total_size *= this->dimensions[i];

  std::vector<float> result {this->current, this->current + total_size};
  return result;
}

void Tensor_gpu::convolution_layer(int output_channel, int kernel, int stride, int padding)
{
	cudnnTensorDescriptor_t input_descriptor;
	checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
	checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                          /*format=*/CUDNN_TENSOR_NHWC,
                        /*dataType=*/CUDNN_DATA_FLOAT,
                      /*batch_size=*/this->dimensions[0],
                        /*channels=*/this->dimensions[1],
                    /*image_height=*/this->dimensions[2],
                     /*image_width=*/this->dimensions[3]));

		cudnnTensorDescriptor_t output_descriptor;
		checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
		checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                          /*format=*/CUDNN_TENSOR_NHWC,
                        /*dataType=*/CUDNN_DATA_FLOAT,
                      /*batch_size=*/this->dimensions[0],
                        /*channels=*/output_channel,
                    /*image_height=*/this->dimensions[2],
                     /*image_width=*/this->dimensions[3]));

	cudnnFilterDescriptor_t kernel_descriptor;
    checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
    checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
                          /*dataType=*/CUDNN_DATA_FLOAT,
                            /*format=*/CUDNN_TENSOR_NCHW,
                      /*out_channels=*/output_channel,
                       /*in_channels=*/this->dimensions[1],
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
  	cudnnGetConvolutionForwardAlgorithm(this->cudnn,
                                      input_descriptor,
                                      kernel_descriptor,
                                      convolution_descriptor,
                                      output_descriptor,
                                      CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                      /*memoryLimitInBytes=*/0,
                                      &convolution_algorithm));


		
		size_t workspace_bytes = 0;
		checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(this->cudnn,
                                                 input_descriptor,
                                                 kernel_descriptor,
                                                 convolution_descriptor,
                                                 output_descriptor,
                                                 convolution_algorithm,
                                                 &workspace_bytes));
		assert(workspace_bytes > 0);

		void* d_workspace{nullptr};
		cudaMalloc(&d_workspace, workspace_bytes);

		int input_image_bytes = this->dimensions[0] * this->dimensions[1] * this->dimensions[2] * this->dimensions[3] * sizeof(float);
		int output_image_bytes = this->dimensions[0] * output_channel * this->dimensions[2] * this->dimensions[3] * sizeof(float);

    float* h_kernel = init_kernel(this->dimensions[1], output_channel, kernel);

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

  	checkCUDNN(cudnnConvolutionForward(this->cudnn,
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
	this->current = h_output;

}

void Tensor_gpu::relu()
{

  int input_image_bytes = this->dimensions[0] * this->dimensions[1] * this->dimensions[2] * this->dimensions[3] * sizeof(float);
  int output_image_bytes = this->dimensions[0] * this->dimensions[1] * this->dimensions[2] * this->dimensions[3] * sizeof(float);

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
                      /*batch_size=*/this->dimensions[0],
                        /*channels=*/this->dimensions[1],
                    /*image_height=*/this->dimensions[2],
                     /*image_width=*/this->dimensions[3]));

  cudnnTensorDescriptor_t output_descriptor;
  checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
  checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                        /*format=*/CUDNN_TENSOR_NHWC,
                      /*dataType=*/CUDNN_DATA_FLOAT,
                    /*batch_size=*/this->dimensions[0],
                      /*channels=*/this->dimensions[1],
                  /*image_height=*/this->dimensions[2],
                   /*image_width=*/this->dimensions[3]));

  const float alpha = 1.0f, beta = 0.0f;
  cudnnActivationDescriptor_t activation_descriptor;
  checkCUDNN(cudnnCreateActivationDescriptor(&activation_descriptor));
  checkCUDNN(cudnnSetActivationDescriptor(activation_descriptor,
                             /*mode=*/CUDNN_ACTIVATION_RELU,
                       /*reluNanOpt=*/CUDNN_PROPAGATE_NAN,
                        /*relu_coef=*/0));

  checkCUDNN(cudnnActivationForward(this->cudnn,
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

  this->current = h_output;
}

float* Tensor_gpu::get_value()
{
    return this->current;
}

__global__ void vecAdd(float *a, float *b, int n)
{
    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;
 
    // Make sure we do not go out of bounds
    if (id < n)
        a[id] = a[id] + b[id];
}

__global__ void vecProduct(float *a, float *b, int n)
{
    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;
 
    // Make sure we do not go out of bounds
    if (id < n)
        a[id] = a[id] * b[id];
}

std::vector<int> Tensor_gpu::get_dim()
{
    return this->dimensions;   
}

void Tensor_gpu::element_product(Tensor_gpu mul)
{
  float* value1 = this->get_value();
  float* value2 = mul.get_value();
  std::vector<int> dims1 = this->get_dim();
  std::vector<int> dims2 = mul.get_dim();

  int total_size = dims1[0];
  for(int i = 1; i < dims1.size(); i++)
    total_size *= dims1[i];
  int bytes = total_size * sizeof(float);

  float* v_a;
  float* v_b;
  cudaMalloc(&v_a, bytes);
  cudaMalloc(&v_b, bytes);
  cudaMemcpy( v_a, value1, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy( v_b, value2, bytes, cudaMemcpyHostToDevice);

  int blockSize, gridSize;
 
  // Number of threads in each thread block
  blockSize = 1024;

  // Number of thread blocks in grid
  gridSize = (int)ceil((float)total_size/blockSize);

  vecProduct<<< gridSize, blockSize >>>(v_a, v_b, total_size);

  cudaMemcpy(value1, v_a, bytes, cudaMemcpyDeviceToHost);

  cudaFree(v_a);
  cudaFree(v_b);

  this->current = &value1[0];

}

void Tensor_gpu::element_add(Tensor_gpu added)
{
  float* value1 = this->get_value();
  float* value2 = added.get_value();
  std::vector<int> dims1 = this->get_dim();
  std::vector<int> dims2 = added.get_dim();

  int total_size = dims1[0];
  for(int i = 1; i < dims1.size(); i++)
    total_size *= dims1[i];
  int bytes = total_size * sizeof(float);

  float* v_a;
  float* v_b;
  cudaMalloc(&v_a, bytes);
  cudaMalloc(&v_b, bytes);
  cudaMemcpy( v_a, value1, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy( v_b, value2, bytes, cudaMemcpyHostToDevice);

  int blockSize, gridSize;
 
  // Number of threads in each thread block
  blockSize = 1024;

  // Number of thread blocks in grid
  gridSize = (int)ceil((float)total_size/blockSize);

  vecAdd<<< gridSize, blockSize >>>(v_a, v_b, total_size);

  cudaMemcpy(value1, v_a, bytes, cudaMemcpyDeviceToHost);

  cudaFree(v_a);
  cudaFree(v_b);

  this->current = &value1[0];
}

