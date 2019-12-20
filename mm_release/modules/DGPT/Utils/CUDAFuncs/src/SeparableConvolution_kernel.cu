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

__global__ void kernel_SeparableConvolution_updateOutput(
	const int n,
	const float* input, const long4 input_size, const long4 input_stride,
	const float* vertical, const long4 vertical_size, const long4 vertical_stride,
	const float* horizontal, const long4 horizontal_size, const long4 horizontal_stride,
	float* output, const long4 output_size, const long4 output_stride
) {
	int intIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (intIndex >= n) {
		return;
	}

	float dblOutput = 0.0;

	int intBatch = ( intIndex / VEC_3(output_size) / VEC_2(output_size) / VEC_1(output_size) ) % VEC_0(output_size);  //b 
	int intDepth = ( intIndex / VEC_3(output_size) / VEC_2(output_size)                      ) % VEC_1(output_size);  //c 
	int intY     = ( intIndex / VEC_3(output_size)                                           ) % VEC_2(output_size);  //y
	int intX     = ( intIndex                                                                ) % VEC_3(output_size);  //x

	for (int intFilterY = 0; intFilterY < VEC_1(vertical_size); intFilterY += 1) {
		for (int intFilterX = 0; intFilterX < VEC_1(vertical_size) ; intFilterX += 1) {
			dblOutput += IDX_4(input, intBatch, intDepth, intY + intFilterY, intX + intFilterX) * IDX_4(vertical, intBatch, intFilterY, intY, intX) * IDX_4(horizontal, intBatch, intFilterX, intY, intX);
		}
	}


	output[intIndex] = dblOutput;
}




void SeparableConvolution_kernel_forward(
	THCState* state,
	THCudaTensor* input,
	THCudaTensor* vertical,
	THCudaTensor* horizontal,
	THCudaTensor* output
) {
	int n = 0;

	n = THCudaTensor_nElement(state, output);
	kernel_SeparableConvolution_updateOutput<<< (n + 512 - 1) / 512, 512, 0, THCState_getCurrentStream(state) >>>(
		n,
		THCudaTensor_data(state, input), get_size(state, input), get_stride(state, input),
		THCudaTensor_data(state, vertical), get_size(state, vertical), get_stride(state, vertical),//make_long4(vertical->size[0], vertical->size[1], vertical->size[2], vertical->size[3]), make_long4(vertical->stride[0], vertical->stride[1], vertical->stride[2], vertical->stride[3]),
		THCudaTensor_data(state, horizontal), get_size(state, horizontal), get_stride(state, horizontal),//make_long4(horizontal->size[0], horizontal->size[1], horizontal->size[2], horizontal->size[3]), make_long4(horizontal->stride[0], horizontal->stride[1], horizontal->stride[2], horizontal->stride[3]),
		THCudaTensor_data(state, output), get_size(state, output), get_stride(state, output)//make_long4(output->size[0], output->size[1], output->size[2], output->size[3]), make_long4(output->stride[0], output->stride[1], output->stride[2], output->stride[3])
	);

	THCudaCheck(cudaGetLastError());
}

__global__ void kernel_SeparableConvolution_updateGradient(
	const int n,
	const float* input, const long4 input_size, const long4 input_stride,
	const float* vertical, const long4 vertical_size, const long4 vertical_stride,
	const float* horizontal, const long4 horizontal_size, const long4 horizontal_stride,
	const float* grad_output, const long4 grad_output_size, const long4 grad_output_stride,
	float* grad_input, const long4 grad_input_size, const long4 grad_input_stride,
	float* grad_vertical, const long4 grad_vertical_size, const long4 grad_vertical_stride,
	float* grad_horizontal, const long4 grad_horizontal_size, const long4 grad_horizontal_stride
) {
	int intIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (intIndex >= n) {
		return;
	}


	int intBatch = ( intIndex / VEC_3(grad_output_size) / VEC_2(grad_output_size)  ) % VEC_0(grad_output_size);
	int intY     = ( intIndex / VEC_3(grad_output_size)                                           ) % VEC_2(grad_output_size);
	int intX     = ( intIndex                                                                ) % VEC_3(grad_output_size);


//calc grad_input
/*
	for (int y = 0; y< VEC_1(vertical_size); y++)
	{
		float total_weight = 0.0;
		for (int x = 0; x < VEC_1(vertical_size); x++)
			total_weight += IDX_4(vertical, intBatch, intDepth,intY+y,intX + x)* IDX_4(horizontal,intBatch,x, intY, intX);
		grad_input[intIndex] = grad_output[intIndex]*total_weight;
	}
*/

//calc grad_vertical 


	for (int y = 0; y< VEC_1(vertical_size); y++)
	{
        float total_weight = 0.0;
		for (int x = 0; x < VEC_1(vertical_size); x++)
		{
		    float tw = 0.0;
		    for (int c = 0; c < VEC_1(grad_output_size); c++ )
		        tw +=  IDX_4(input,intBatch, c,intY+y,intX + x)*IDX_4(grad_output,intBatch, c,intY,intX );

		    total_weight += tw* IDX_4(horizontal,intBatch,x, intY, intX);
        }

        IDX_4(grad_vertical ,intBatch,y , intY, intX) = total_weight;
	}



//calc grad_horizontal 
	for (int x = 0; x< VEC_1(vertical_size); x++)
	{
        float total_weight = 0.0;
		for (int y = 0; y < VEC_1(vertical_size); y++)
		{
		    float tw = 0.0;
		    for (int c = 0; c < VEC_1(grad_output_size); c++ )
		        tw +=  IDX_4(input,intBatch, c,intY+y,intX + x)*IDX_4(grad_output,intBatch, c,intY,intX );

		    total_weight += tw* IDX_4(vertical,intBatch,y, intY, intX);
        }

        IDX_4(grad_horizontal ,intBatch,x , intY, intX) = total_weight;
	}


}
//  output = input * v * h
//  grad_v = input * h * grad_out

void SeparableConvolution_kernel_backward(
	THCState* state,
	THCudaTensor* input,
	THCudaTensor* vertical,
	THCudaTensor* horizontal,
	THCudaTensor* grad_output,
	THCudaTensor* grad_input,
	THCudaTensor* grad_vertical,
	THCudaTensor* grad_horizontal

) {
	int n = 0;

	//n = THCudaTensor_nElement(state, grad_output);
	n = THCudaTensor_size(state, grad_output,0) *  THCudaTensor_size(state, grad_output,2) * THCudaTensor_size(state, grad_output,3);
	//n = grad_output->size[0]* grad_output->size[2]*grad_output->size[3];
	kernel_SeparableConvolution_updateGradient<<< (n + 512 - 1) / 512, 512, 0, THCState_getCurrentStream(state) >>>(
		n,
		THCudaTensor_data(state, input), get_size(state, input), get_stride(state, input),//make_long4(input->size[0], input->size[1], input->size[2], input->size[3]), make_long4(input->stride[0], input->stride[1], input->stride[2], input->stride[3]),
		THCudaTensor_data(state, vertical), get_size(state, vertical), get_stride(state, vertical),//make_long4(vertical->size[0], vertical->size[1], vertical->size[2], vertical->size[3]), make_long4(vertical->stride[0], vertical->stride[1], vertical->stride[2], vertical->stride[3]),
		THCudaTensor_data(state, horizontal), get_size(state, horizontal), get_stride(state, horizontal),//make_long4(horizontal->size[0], horizontal->size[1], horizontal->size[2], horizontal->size[3]), make_long4(horizontal->stride[0], horizontal->stride[1], horizontal->stride[2], horizontal->stride[3]),
		THCudaTensor_data(state, grad_output), get_size(state, grad_output), get_stride(state, grad_output),//make_long4(grad_output->size[0], grad_output->size[1], grad_output->size[2], grad_output->size[3]), make_long4(grad_output->stride[0], grad_output->stride[1], grad_output->stride[2], grad_output->stride[3]),
		THCudaTensor_data(state, grad_input), get_size(state, grad_input), get_stride(state, grad_input),//make_long4(grad_input->size[0], grad_input->size[1], grad_input->size[2], grad_input->size[3]), make_long4(grad_input->stride[0], grad_input->stride[1], grad_input->stride[2], grad_input->stride[3]),
		THCudaTensor_data(state, grad_vertical), get_size(state, grad_vertical), get_stride(state, grad_vertical),//make_long4(grad_vertical->size[0], grad_vertical->size[1], grad_vertical->size[2], grad_vertical->size[3]), make_long4(grad_vertical->stride[0], grad_vertical->stride[1], grad_vertical->stride[2], grad_vertical->stride[3]),
		THCudaTensor_data(state, grad_horizontal), get_size(state, grad_horizontal), get_stride(state, grad_horizontal)//make_long4(grad_horizontal->size[0], grad_horizontal->size[1], grad_horizontal->size[2], grad_horizontal->size[3]), make_long4(grad_horizontal->stride[0], grad_horizontal->stride[1], grad_horizontal->stride[2], grad_horizontal->stride[3])

	);

	THCudaCheck(cudaGetLastError());
}

#ifdef __cplusplus
	}
#endif
