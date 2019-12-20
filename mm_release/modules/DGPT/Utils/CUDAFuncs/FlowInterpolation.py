import torch
import DGPTCUDA
# from DGPT.Utils.CUDAFuncs._ext import cunnex

# void FlowBlur_kernel_forward(
#     THCState * state,
#     THCudaTensor * in_flow,
#     THCudaTensor * in_mask,
#     THCudaTensor * out_flow,
#     THCudaTensor * out_mask,
#     THCudaTensor * tmp1_flow,
#     THCudaTensor * tmp1_mask,
#     THCudaTensor * tmp2_flow,
#     THCudaTensor * tmp2_mask,
# )
# // flow shape HxWx2
# // mask shape HxW

class FlowInterpolation(torch.autograd.Function):
    def __init__(self):
        super(FlowInterpolation, self).__init__()

    # @staticmethod
    def forward(self, input_all, mask_all):
        intBatches = input_all.size(0)
        intInputDepth = input_all.size(1)
        intInputHeight = input_all.size(2)
        intInputWidth = input_all.size(3)

        assert (input_all.is_contiguous() == True)

        out_all = torch.zeros_like(input_all)

        for i in range(intBatches):
            input = input_all[i]
            mask = mask_all[i].squeeze(0)

            output = input.new().resize_(intInputHeight, intInputWidth, 2).zero_()
            out_mask = mask.new().resize_(intInputHeight, intInputWidth).zero_()

            tmp1 = input.new().resize_(intInputHeight, intInputWidth, 2).zero_()
            tmpm1 = mask.new().resize_(intInputHeight, intInputWidth).zero_()

            tmp0 = input.new().resize_(intInputHeight, intInputWidth, 2).zero_()
            tmpm0 = mask.new().resize_(intInputHeight, intInputWidth).zero_()

            if input.is_cuda == True:
                DGPTCUDA.FlowBlur_cuda_forward(
                    input,
                    mask,
                    output,
                    out_mask,
                    tmp0,
                    tmpm0,
                    tmp1,
                    tmpm1
                )

            elif input.is_cuda == False:
                raise NotImplementedError()

            out_all[i]=output

        return out_all

    def backward(self, grad_out):
        raise NotImplementedError()





