int SeparableConvolution_cuda_forward(
	at::Tensor input,
	at::Tensor vertical,
	at::Tensor horizontal,
	at::Tensor output
);

int SeparableConvolution_cuda_backward(
	at::Tensor input,
	at::Tensor vertical,
	at::Tensor horizontal,
	at::Tensor grad_out,
	at::Tensor input_g,
	at::Tensor vertical_g,
	at::Tensor horizontal_g);


int FlowMover_cuda_forward(
	at::Tensor input,  //input image , in the shape of B, 3, H, W , H and W should be at least multiples of 16
	at::Tensor yx, //flo , B , H, W , 2
	at::Tensor output, // output image, B , 3 , H , W
	at::Tensor weight  // weight, B , 1 , H , W
);

int GaussianBlur_cuda_forward(
//	THCState* state,
	at::Tensor input,
	at::Tensor temp,
	at::Tensor output,
	float  sigma
);

int EigenAnalysis_cuda_forward(
//    THCState* state,
	at::Tensor input,
	at::Tensor output
);

int EigenAnalysis_cuda_backward(
//    THCState* state,
	at::Tensor input,
	at::Tensor grad_output,
	at::Tensor grad_input
);

int FlowChecker_cuda_forward(
	at::Tensor left,  //left image , in the shape of B, 3, H, W , H and W should be at least multiples of 16
	at::Tensor lflow, //flo , B , H, W , 2
	at::Tensor right, // right image, B , 3 , H , W
	at::Tensor rflow  // weight, B , 1 , H , W
);

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
);
