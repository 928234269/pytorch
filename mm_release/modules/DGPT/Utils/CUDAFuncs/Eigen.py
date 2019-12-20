import torch
# import _ext.cunnex
# from DGPT.Utils.CUDAFuncs._ext import cunnex
# from DGPT.Utils.CUDAFuncs.DGPTCUDA import EigenAnalysis_cuda_forward
import DGPTCUDA

class Eigen_CUDA(torch.autograd.Function):
    def __init__(self):
        super(Eigen_CUDA, self).__init__()

    # @staticmethod
    def forward(self, input):
        intBatches = input.size(0)
        intInputDepth = input.size(1)
        intInputHeight = input.size(2)
        intInputWidth = input.size(3)

        assert (input.is_contiguous() == True)

        self.save_for_backward(input)
        # self.input = input.clone()
        output = input.new().resize_(intBatches, 4, intInputHeight, intInputWidth).zero_()
        # temp = input.new().resize_(intBatches, intInputDepth, intInputHeight, intInputWidth).zero_()

        if input.is_cuda == True:
            DGPTCUDA.EigenAnalysis_cuda_forward(
                input,
                output,
            )

        elif input.is_cuda == False:
            raise NotImplementedError()

        return output

    # @staticmethod
    def backward(self, grad_out):
        input = self.saved_tensors
        input = input[0]

        # print(input.size())

        # b, c, h, w = input.size()
        grad_in = torch.zeros_like(input)

        DGPTCUDA.EigenAnalysis_cuda_backward(
            input,
            grad_out,
            grad_in
        )

        return grad_in



if __name__ == '__main__':
    from torch.autograd import gradcheck
    dev = torch.device("cuda")

    grad = torch.rand(1,2,2,2)

    xx = grad[:,0:1]*grad[:,0:1]
    xy = grad[:,0:1]*grad[:,1:2]
    yy = grad[:, 1:2] * grad[:, 1:2]

    input = torch.cat((xx,xy,yy),1).to(dev)

    # input = torch.rand(1,3,2,2).to(dev)
    input.requires_grad=True

    eig = Eigen_CUDA()
    #
    # out = eig(input)
    # print(out)
    # out.mean().backward()

    import torch

    # l = torch.nn.Linear(5,5)
    #
    # lin = torch.rand(1,5)#.double()
    # lin.requires_grad = True
    #
    # res = gradcheck(l, (lin,), eps=1e-3)

    print('checking...')
    res = gradcheck(eig, (input,), eps=1e-3)
    print(res)

# end

# def backward(self, gradOutput):
# 	raise NotImplementedError() # BACKPROPAGATION NOT IMPLEMENTED
# end
# end
