#include <THC.h>
#include <THCGeneral.h>



#include "utils.h"

#define VEC_0(ARRAY) ((ARRAY).x)
#define VEC_1(ARRAY) ((ARRAY).y)
#define VEC_2(ARRAY) ((ARRAY).z)
#define VEC_3(ARRAY) ((ARRAY).w)

#define IDX_1(ARRAY, X)          ((ARRAY)[((X) * (ARRAY##_stride.x))])
#define IDX_2(ARRAY, X, Y)       ((ARRAY)[((X) * (ARRAY##_stride.x)) + ((Y) * (ARRAY##_stride.y))])
#define IDX_3(ARRAY, X, Y, Z)    ((ARRAY)[((X) * (ARRAY##_stride.x)) + ((Y) * (ARRAY##_stride.y)) + ((Z) * (ARRAY##_stride.z))])
#define IDX_4(ARRAY, X, Y, Z, W) ((ARRAY)[((X) * (ARRAY##_stride.x)) + ((Y) * (ARRAY##_stride.y)) + ((Z) * (ARRAY##_stride.z)) + ((W) * (ARRAY##_stride.w))])

#ifdef __cplusplus
	extern "C" {
#endif

__global__ void kernel_EigenAnalysis_updateOutput(
	const int n,
	const float* input, const long4 input_size, const long4 input_stride,
	float* output, const long4 output_size, const long4 output_stride
) {
	int intIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (intIndex >= n) {
		return;
	}


	int intBatch = ( intIndex / VEC_3(output_size) / VEC_2(output_size)  ) % VEC_0(output_size);  //b
	int intY     = ( intIndex / VEC_3(output_size)                                           ) % VEC_2(output_size);  //y
	int intX     = ( intIndex                                                                ) % VEC_3(output_size);  //x

    int offset = intBatch*input_stride.x + intY*input_stride.z + intX*input_stride.w;

    float d = input[offset];
    offset += input_stride.y;
    float e = input[offset];
    offset += input_stride.y;
    float g = input[offset];

    float delta = sqrt(4* e*e + (d-g)*(d-g));
    float lamda0 = (d+g + delta)/2;
    float lamda1 = lamda0 - delta;

    float L0 = sqrt(e*e+(lamda0-d)*(lamda0-d));
    float L1 = sqrt(e*e+(lamda1-d)*(lamda1-d));
    const float epsilon = 0.00000001;

    float x0= e*lamda0/(L0+epsilon);
    float y0= (lamda0-d)*lamda0/(L0+epsilon);

    float x1= e*lamda1/(L1+epsilon);
    float y1= (lamda1-d)*lamda1/(L1+epsilon);

    offset= intBatch*output_stride.x + intY*output_stride.z + intX*output_stride.w;
    output[offset]=x0;offset+=output_stride.y;
    output[offset]=y0;offset+=output_stride.y;
    output[offset]=x1;offset+=output_stride.y;
    output[offset]=y1;
}


//input: [batch, 3, h, w] : 0:x*x, 1:x*y, 2:y*y
//output:[batch, 4, h, w] : 0:x_big, 1:y_big, 2: x_small, 3:y_small

void EigenAnalysis_kernel_forward(
	at::Tensor input,
	at::Tensor output
) {
	int n = input.size(0) * input.size(2)* input.size(3);

	kernel_EigenAnalysis_updateOutput<<< (n + 512 - 1) / 512, 512 >>>
	(
		n,
		TENSOR_INFO(input),
		TENSOR_INFO(output)

//		input.data(),make_long4(input.size(0),input.size(1),input.size(2),input.size(3)), make_long4(input.stride(0),input.stride(1),input.stride(2),input.stride(3)),
//		output.data(),make_long4(output.size(0),output.size(1),output.size(2),output.size(3)), make_long4(output.stride(0),output.stride(1),output.stride(2),output.stride(3))
	);




	THCudaCheck(cudaGetLastError());
}


__global__ void kernel_EigenAnalysis_updateGradient(
	const int n,
	const float* input, const long4 input_size, const long4 input_stride,
	const float* grad_output, const long4 grad_output_size, const long4 grad_output_stride,
	float* grad_input, const long4 grad_input_size, const long4 grad_input_stride
) {
	int intIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (intIndex >= n) {
		return;
	}
    const float epsilon = 0.00000000;


	int intBatch = ( intIndex / VEC_3(input_size) / VEC_2(input_size)  ) % VEC_0(input_size);  //b
	int intY     = ( intIndex / VEC_3(input_size)                                           ) % VEC_2(input_size);  //y
	int intX     = ( intIndex                                                                ) % VEC_3(input_size);  //x

    int offset = intBatch*input_stride.x + intY*input_stride.z + intX*input_stride.w;

    float d = input[offset];
    offset += input_stride.y;
    float e = input[offset];
    offset += input_stride.y;
    float g = input[offset];

    float delta = sqrt(4* e*e + (d-g)*(d-g))+epsilon;
    float lamda0 = (d+g + delta)/2;
    float lamda1 = lamda0 - delta;




    float L0 = sqrt(e*e+(lamda0-d)*(lamda0-d))+epsilon;
    float L1 = sqrt(e*e+(lamda1-d)*(lamda1-d))+epsilon;

    if (L0 == 0.0 || L1 == 0.0 || delta == 0.0 )
    {
          offset= intBatch*grad_input_stride.x + intY*grad_input_stride.z + intX*grad_input_stride.w;


        grad_input[offset]=0.0f;offset+=grad_input_stride.y;
        grad_input[offset]=0.0f;offset+=grad_input_stride.y;
        grad_input[offset]=0.0f;
        return ;
    }


    float d_lamda0_dd = (1+ (d-g)/delta)/2;
    float d_lamda1_dd = (1- (d-g)/delta)/2;

    float d_lamda0_dg = d_lamda1_dd;
    float d_lamda1_dg = d_lamda0_dd;

    float d_lamda0_de = + 2* e/ delta;
    float d_lamda1_de = - d_lamda0_de;

    offset = intBatch*grad_output_stride.x + intY*grad_output_stride.z + intX*grad_output_stride.w;
    float grad_x0 = grad_output[offset];offset+=grad_output_stride.y;
    float grad_y0 = grad_output[offset];offset+=grad_output_stride.y;
    float grad_x1 = grad_output[offset];offset+=grad_output_stride.y;
    float grad_y1 = grad_output[offset];



    float grad_e = (lamda0/L0+ e*d_lamda0_de/L0 - e*lamda0/L0/L0/L0*(e+(lamda0-d)*d_lamda0_de))*grad_x0;
    grad_e += (lamda1/L1+ e*d_lamda1_de/L1 - e*lamda1/L1/L1/L1*(e+(lamda1-d)*d_lamda1_de))*grad_x1;
    grad_e += ((2*lamda0- d)/L0* d_lamda0_de - (lamda0-d)*lamda0*(e+(lamda0-d)*d_lamda0_de)/L0/L0/L0)*grad_y0;
    grad_e += ((2*lamda1- d)/L1* d_lamda1_de - (lamda1-d)*lamda1*(e+(lamda1-d)*d_lamda1_de)/L1/L1/L1)*grad_y1;

    float grad_d = (e/L0*d_lamda0_dd + e*lamda0/L0/L0/L0*(lamda0-d)*(1-d_lamda0_dd))*grad_x0;
    grad_d += (e/L1*d_lamda1_dd + e*lamda1/L1/L1/L1*(lamda1-d)*(1-d_lamda1_dd))*grad_x1;
    grad_d += (((d_lamda0_dd -1)*lamda0 + (lamda0-d)*d_lamda0_dd)/L0 + (lamda0-d)*(lamda0-d)*lamda0*(1-d_lamda0_dd)/L0/L0/L0)
        *grad_y0;
    grad_d += (((d_lamda1_dd -1)*lamda1 + (lamda1-d)*d_lamda1_dd)/L1 + (lamda1-d)*(lamda1-d)*lamda1*(1-d_lamda1_dd)/L1/L1/L1)
        *grad_y1;


    float grad_g = (e/L0*d_lamda0_dg + e*lamda0/L0/L0/L0*(lamda0-d)*(-d_lamda0_dg))*grad_x0;
    grad_g += (e/L1*d_lamda1_dg + e*lamda1/L1/L1/L1*(lamda1-d)*(-d_lamda1_dg))*grad_x1;
    grad_g += ((d_lamda0_dg*lamda0 + (lamda0-d)*d_lamda0_dg)/L0 + (lamda0-d)*(lamda0-d)*lamda0*(-d_lamda0_dg)/L0/L0/L0)
        *grad_y0;
    grad_g += ((d_lamda1_dg*lamda1 + (lamda1-d)*d_lamda1_dg)/L1 + (lamda1-d)*(lamda1-d)*lamda1*(-d_lamda1_dg)/L1/L1/L1)
        *grad_y1;


    offset= intBatch*grad_input_stride.x + intY*grad_input_stride.z + intX*grad_input_stride.w;


    grad_input[offset]=grad_d;offset+=grad_input_stride.y;
    grad_input[offset]=grad_e;offset+=grad_input_stride.y;
    grad_input[offset]=grad_g;


}
//  output = input * v * h
//  grad_v = input * h * grad_out
/*
void EigenAnalysis_kernel_backward(
	THCudaTensor* input,
	THCudaTensor* grad_output,
	THCudaTensor* grad_input

) {
	int n = 0;

    n = THCudaTensor_size(state, grad_output,0)*THCudaTensor_size(state, grad_output,2)*THCudaTensor_size(state, grad_output,3);
	//n = grad_output->size[0]* grad_output->size[2]*grad_output->size[3];
	kernel_EigenAnalysis_updateGradient<<< (n + 512 - 1) / 512, 512, 0, THCState_getCurrentStream(state) >>>(
		n,
		THCudaTensor_data(state, input), get_size(state, input), get_stride(state, input), //make_long4(input->size[0], input->size[1], input->size[2], input->size[3]), make_long4(input->stride[0], input->stride[1], input->stride[2], input->stride[3]),
		THCudaTensor_data(state, grad_output), get_size(state, grad_output), get_stride(state, grad_output), //make_long4(grad_output->size[0], grad_output->size[1], grad_output->size[2], grad_output->size[3]), make_long4(grad_output->stride[0], grad_output->stride[1], grad_output->stride[2], grad_output->stride[3]),
		THCudaTensor_data(state, grad_input), get_size(state, grad_input), get_stride(state, grad_input) //make_long4(grad_input->size[0], grad_input->size[1], grad_input->size[2], grad_input->size[3]), make_long4(grad_input->stride[0], grad_input->stride[1], grad_input->stride[2], grad_input->stride[3])

	);

	THCudaCheck(cudaGetLastError());
}
*/
void EigenAnalysis_kernel_backward(

	at::Tensor  input,
	at::Tensor grad_output,
	at::Tensor grad_input
)
{
	int n = grad_output.size(0) * grad_output.size(2)* grad_output.size(3);

	kernel_EigenAnalysis_updateGradient<<< (n + 512 - 1) / 512, 512 >>>
	(
		n,
		TENSOR_INFO(input),
		TENSOR_INFO(grad_output),
		TENSOR_INFO(grad_input)

//		input.data(),make_long4(input.size(0),input.size(1),input.size(2),input.size(3)), make_long4(input.stride(0),input.stride(1),input.stride(2),input.stride(3)),
//		grad_output.data(),make_long4(grad_output.size(0),grad_output.size(1),grad_output.size(2),grad_output.size(3)), make_long4(grad_output.stride(0),grad_output.stride(1),grad_output.stride(2),grad_output.stride(3)),
//		grad_input.data(),make_long4(grad_input.size(0),grad_input.size(1),grad_input.size(2),grad_input.size(3)), make_long4(grad_input.stride(0),grad_input.stride(1),grad_input.stride(2),grad_input.stride(3))
	);

	THCudaCheck(cudaGetLastError());
}

#ifdef __cplusplus
	}
#endif
