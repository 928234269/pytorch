#ifdef __cplusplus
	extern "C" {
#endif

void SeparableConvolution_kernel_forward(

	at::Tensor input,
	at::Tensor vertical,
	at::Tensor horizontal,
	at::Tensor output
);

void SeparableConvolution_kernel_backward(
	at::Tensor input,
	at::Tensor vertical,
	at::Tensor horizontal,
	at::Tensor grad_output,
	at::Tensor grad_input,
	at::Tensor grad_vertical,
	at::Tensor grad_horizontal

);

void FlowMover_kernel_forward(
	at::Tensor input,  //input image , in the shape of B, 3, H, W , H and W should be at least multiples of 16
	at::Tensor yx, //flo , B , H, W , 2
	at::Tensor output, // output image, B , 3 , H , W
	at::Tensor weight  // weight, B , 1 , H , W
);

void FlowChecker_kernel_forward(
	at::Tensor left,  //left image , in the shape of B, 3, H, W , H and W should be at least multiples of 16
	at::Tensor lflow, //flo , B , H, W , 2
	at::Tensor right, // right image, B , 3 , H , W
	at::Tensor rflow  // weight, B , 1 , H , W
);

void GaussianBlur_kernel_forward(
	at::Tensor input,
	at::Tensor temp,
	at::Tensor output,
	float  sigma
);

void EigenAnalysis_kernel_forward(
	at::Tensor input,
	at::Tensor output
//	at::Tensor input,
//	at::Tensor output
);

void EigenAnalysis_kernel_backward(
//    THCState* state,
//	at::Tensor input,
//	at::Tensor grad_output,
//	at::Tensor grad_input

    at::Tensor input,
    at::Tensor grad_output,
    at::Tensor grad_input
);

void FlowBlur_kernel_forward(
	at::Tensor in_flow,
	at::Tensor in_mask,
	at::Tensor out_flow,
	at::Tensor out_mask,
	at::Tensor tmp1_flow,
	at::Tensor tmp1_mask,
	at::Tensor tmp2_flow,
	at::Tensor tmp2_mask
);

#ifdef __cplusplus
	}
#endif
