import torch
import DGPTCUDA
# from ._ext import cunnex

class FlowMover(torch.autograd.Function):
    def __init__(self):
        super(FlowMover, self).__init__()

    # end

    def forward(self, input, flow):
        intBatches = input.size(0)
        intInputDepth = input.size(1)
        intInputHeight = input.size(2)
        intInputWidth = input.size(3)

        assert (input.is_contiguous() == True)
        assert (flow.is_contiguous() == True)

        output = input.new().resize_(intBatches, intInputDepth, intInputHeight, intInputWidth).zero_()
        weight = input.new().resize_(intBatches, 1, intInputHeight, intInputWidth).fill_(1e-8)

        if input.is_cuda == True:
            DGPTCUDA.FlowMover_cuda_forward(
                input,
                flow,
                output,
                weight
            )

        elif input.is_cuda == False:
            raise NotImplementedError()

        output = output.div(weight)
                 # - weight.eq(1e-8).float()
        return output , weight


# end

# def backward(self, gradOutput):
# 	raise NotImplementedError() # BACKPROPAGATION NOT IMPLEMENTED
# end
# end
