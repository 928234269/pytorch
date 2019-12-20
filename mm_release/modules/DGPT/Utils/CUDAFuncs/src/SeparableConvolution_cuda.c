#include <THC.h>
#include <THCGeneral.h>
#include <torch/torch.h>

#include <vector>

#include "SeparableConvolution_kernel.h"

extern THCState* state;

int SeparableConvolution_cuda_forward(
	at::Tensor input,
	at::Tensor vertical,
	at::Tensor horizontal,
	at::Tensor output
) {
	SeparableConvolution_kernel_forward(
		state,
		input,
		vertical,
		horizontal,
		output
	);

	return 1;
}

int SeparableConvolution_cuda_backward(
	at::Tensor input,
	at::Tensor vertical,
	at::Tensor horizontal,
	at::Tensor grad_out,
	at::Tensor input_g,
	at::Tensor vertical_g,
	at::Tensor horizontal_g)
{
	SeparableConvolution_kernel_backward(
		state,
		input,
		vertical,
		horizontal,
		grad_out,
		input_g,
		vertical_g,
		horizontal_g
	);

	return 1;
}


int FlowMover_cuda_forward(
	at::Tensor input,  //input image , in the shape of B, 3, H, W , H and W should be at least multiples of 16
	at::Tensor yx, //flo , B , H, W , 2
	at::Tensor output, // output image, B , 3 , H , W
	at::Tensor weight  // weight, B , 1 , H , W
)
{
    FlowMover_kernel_forward(
        state,
        input,
        yx,
        output,
        weight
        );

    return 1;
}

int FlowChecker_cuda_forward(
	at::Tensor left,  //left image , in the shape of B, 3, H, W , H and W should be at least multiples of 16
	at::Tensor lflow, //flo , B , H, W , 2
	at::Tensor right, // right image, B , 3 , H , W
	at::Tensor rflow  // weight, B , 1 , H , W
)
{
    FlowChecker_kernel_forward(
        state,
        left,
        lflow,
        right,
        rflow
        );

    return 1;
}

int EigenAnalysis_cuda_forward(
//    THCState* state,
	at::Tensor input,
	at::Tensor output
){
    EigenAnalysis_kernel_forward(state, input, output);
    return 1;
}


int EigenAnalysis_cuda_backward(
//    THCState* state,
	at::Tensor input,
	at::Tensor grad_output,
	at::Tensor grad_input
){
    EigenAnalysis_kernel_backward(state, input, grad_output, grad_input);
    return 1;
}

int GaussianBlur_cuda_forward(
//	THCState* state,
	at::Tensor input,
	at::Tensor temp,
	at::Tensor output,
	float  sigma
){
    GaussianBlur_kernel_forward(
        state, input, temp, output, sigma
    );
    return 1;
}

int FlowBlur_cuda_forward(
//    THCState* state,
	at::Tensor in_flow,
	at::Tensor in_mask,
	at::Tensor out_flow,
	at::Tensor out_mask,
	at::Tensor tmp1_flow,
	at::Tensor tmp1_mask,
	at::Tensor tmp2_flow,
	at::Tensor tmp2_mask
){
    FlowBlur_kernel_forward(
        state, in_flow, in_mask, out_flow, out_mask, tmp1_flow, tmp1_mask, tmp2_flow, tmp2_mask
    );
    return 1;
}
