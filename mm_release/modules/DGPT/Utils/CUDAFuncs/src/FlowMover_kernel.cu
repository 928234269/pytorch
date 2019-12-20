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



__global__ void kernel_FlowMover_updateOutput(
	const int n, 
	const float * input,  const long4 input_size, const long4 input_stride,
	const float * yx, const long4 yx_size, const long4 yx_stride,
	float * output,  const long4 output_size, const long4 output_stride,
	float * weight, const long4 weight_size, const long4 weight_stride
) 

{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= n)
		return;
	
	



	int intBatch = ( idx / VEC_3(output_size) / VEC_2(output_size)  ) % VEC_0(output_size);
	int intY     = ( idx / VEC_3(output_size) ) % VEC_2(output_size);
	int intX     = ( idx ) % VEC_3(output_size);
	
	 
	//int dX = (int) yx[intIndex].x; //? .x or .y ; need to test
	//int dY = (int) yx[intIndex].y; //? .y or .x
	int dX = (int) (IDX_4(yx, intBatch, intY, intX, 0) +0.5) +intX;
	int dY = (int) (IDX_4(yx, intBatch, intY, intX, 1) +0.5) +intY;

    if (dX < 0 || dX >= output_size.w || dY < 0 || dY >=output_size.z)
        return;

	int outIndex = intBatch * output_stride.x + dY*output_stride.z + dX;
	int inputIndex = intBatch* input_stride.x + intY * input_stride.z + intX;

    for (int i = 0; i< output_size.y; i++)
            atomicAdd(output+ outIndex + output_stride.y*i, input[inputIndex + input_stride.y*i]);
    //atomicAdd(output+ outIndex+output_stride.y, input[inputIndex+input_stride.y]);
    //atomicAdd(output+ outIndex+output_stride.y*2, input[inputIndex+input_stride.y*2]);

    atomicAdd(weight+ intBatch* weight_stride.x + dY* weight_stride.z + dX, 1.0);

/*
	output[outIndex] += input[inputIndex];
	output[outIndex+output_stride.y] += input[inputIndex+input_stride.y];
	output[outIndex+output_stride.y*2] += input[inputIndex+input_stride.y*2];

    weight[intBatch* weight_stride.x + dY* weight_stride.z + dX] ++;
*/


}



void FlowMover_kernel_forward(
	//THCState* state,
	at::Tensor input,  //input image , in the shape of B, 3, H, W , H and W should be at least multiples of 16
	at::Tensor yx, //flo , B , H, W , 2
	at::Tensor output, // output image, B , 3 , H , W
	at::Tensor weight  // weight, B , 1 , H , W
) 
//output should be filled with zero 
//weight should be filled with 1e-8
//after calling FlowMover, repeat weight to B , 3 ,H ,W , output /= weight 


{
	int n = 0;


    int fuck = output.size(1);// THCudaTensor_size(state,output,1);
	n = output.size(0)*output.size(2)*output.size(3);//THCudaTensor_nElement(state, output)/fuck;

	kernel_FlowMover_updateOutput<<< (n + 512 - 1) / 512, 512 >>>(
	//kernel_FlowMover_updateOutput<<< (n + 512 - 1) / 512, 512, 0, THCState_getCurrentStream(state) >>>(
		n,
		TENSOR_INFO(input),
		TENSOR_INFO(yx),
		TENSOR_INFO(output),
		TENSOR_INFO(weight)
		//THCudaTensor_data(state, input), get_size(state, input),get_stride(state, input),
		//THCudaTensor_data(state, yx), get_size(state, yx), get_stride(state, yx), //make_long4(yx->size[0], yx->size[1], yx->size[2], yx->size[3]), make_long4(yx->stride[0], yx->stride[1], yx->stride[2], yx->stride[3]),
		//THCudaTensor_data(state, output), get_size(state, output),get_stride(state, output), //make_long4(output->size[0], output->size[1], output->size[2], output->size[3]), make_long4(output->stride[0], output->stride[1], output->stride[2], output->stride[3]),
		//THCudaTensor_data(state, weight), get_size(state, weight),get_stride(state, weight) //make_long4(weight->size[0], weight->size[1], weight->size[2], weight->size[3]), make_long4(weight->stride[0], weight->stride[1], weight->stride[2], weight->stride[3])

	);

	THCudaCheck(cudaGetLastError());
}
#ifdef __cplusplus
	}
#endif

