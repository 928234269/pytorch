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



__global__ void kernel_FlowChecker_updateOutput(
	const int n,
	const float * left,  const long4 left_size, const long4 left_stride,
    float * lflow, const long4 lflow_size, const long4 lflow_stride,
    	const float * right,  const long4 right_size, const long4 right_stride,
    float * rflow, const long4 rflow_size, const long4 rflow_stride
)

{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= n)
		return;





	int intBatch = ( idx / VEC_3(left_size) / VEC_2(left_size)  ) % VEC_0(left_size);
	int intY     = ( idx / VEC_3(left_size) ) % VEC_2(left_size);
	int intX     = ( idx ) % VEC_3(left_size);

    int yxoffset = intBatch * lflow_stride.x + intY * lflow_stride.y + intX * lflow_stride.z;

    int lX = (int)( lflow[yxoffset] + 0.5 + intX);
    int lY = (int)( lflow[yxoffset+1] + 0.5 + intY);

    int rX = (int)( rflow[yxoffset] + 0.5 + intX);
    int rY = (int)( rflow[yxoffset+1] + 0.5 + intY);

    int srcIndex = intBatch* left_stride.x + intY * left_stride.z + intX;

    if (lX >=0 && lX < left_size.w && lY >=0 && lY < left_size.z)
    {
        int dstIndex = intBatch* left_stride.x + lY * left_stride.z + lX;
        int i ;
        for (i = 0; i < 3; i++, srcIndex+= left_stride.y, dstIndex+= left_stride.y)
        {
            float src = left[srcIndex]+ 0.03; //allow some noise , 0.05
            float dst = right[dstIndex]+0.03;
            if (src > dst *1.3 || dst > src * 1.3)
                break;
        }
        if (i< 3)
            lflow[yxoffset] =  - 100000;

    }

    srcIndex = intBatch* left_stride.x + intY * left_stride.z + intX;
    if (rX >=0 && rX < left_size.w && rY >=0 && rY < left_size.z)
    {
        int dstIndex = intBatch* left_stride.x + rY * left_stride.z + rX;
        int i ;
        for (i = 0; i < 3; i++, srcIndex+= left_stride.y, dstIndex+= left_stride.y)
        {
            float src = right[srcIndex]+ 0.03; //allow some noise , 0.05
            float dst = left[dstIndex]+0.03;
            if (src > dst *1.3 || dst > src * 1.3)
                break;
        }
        if (i< 3)
            rflow[yxoffset] =  - 100000;

    }
}



void FlowChecker_kernel_forward(
	THCState* state,
	THCudaTensor* left,  //left image , in the shape of B, 3, H, W , H and W should be at least multiples of 16
	THCudaTensor* lflow, //flo , B , H, W , 2
	THCudaTensor* right, // right image, B , 3 , H , W
	THCudaTensor* rflow  // weight, B , 1 , H , W
)
//output should be filled with zero
//weight should be filled with 1e-8
//after calling FlowMover, repeat weight to B , 3 ,H ,W , output /= weight


{
	int n = 0;

	n = THCudaTensor_nElement(state, left)/THCudaTensor_size(state, left, 1);   //left->size[1];

	kernel_FlowChecker_updateOutput<<< (n + 512 - 1) / 512, 512, 0, THCState_getCurrentStream(state) >>>(
		n,
		THCudaTensor_data(state, left)
		    , get_size(state, left) //make_long4(left->size[0], left->size[1], left->size[2], left->size[3])
		    , get_stride(state, left), //make_long4(left->stride[0], left->stride[1], left->stride[2], left->stride[3]),
		THCudaTensor_data(state, lflow)
		    , get_size(state, lflow) //make_long4(left->size[0], left->size[1], left->size[2], left->size[3])
		    , get_stride(state, lflow), //make_long4(left->stride[0], left->stride[1], left->stride[2], left->stride[3]),
		    //, make_long4(lflow->size[0], lflow->size[1], lflow->size[2], lflow->size[3])
		    //, make_long4(lflow->stride[0], lflow->stride[1], lflow->stride[2], lflow->stride[3]),

		THCudaTensor_data(state, right)
		    , get_size(state, right) //make_long4(left->size[0], left->size[1], left->size[2], left->size[3])
		    , get_stride(state, right), //make_long4(left->stride[0], left->stride[1], left->stride[2], left->stride[3]),

		    //, make_long4(right->size[0], right->size[1], right->size[2], right->size[3])
		    //, make_long4(right->stride[0], right->stride[1], right->stride[2], right->stride[3]),
		THCudaTensor_data(state, rflow)
		    , get_size(state, rflow) //make_long4(left->size[0], left->size[1], left->size[2], left->size[3])
		    , get_stride(state, rflow) //make_long4(left->stride[0], left->stride[1], left->stride[2], left->stride[3]),

		    //, make_long4(rflow->size[0], rflow->size[1], rflow->size[2], rflow->size[3])
		    //, make_long4(rflow->stride[0], rflow->stride[1], rflow->stride[2], rflow->stride[3])



	);

	THCudaCheck(cudaGetLastError());
}
#ifdef __cplusplus
	}
#endif