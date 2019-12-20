import torch
# import _ext.cunnex
# from DGPT.Utils.CUDAFuncs._ext import cunnex
import DGPTCUDA

class GaussianBlur_CUDA(torch.autograd.Function):
    def __init__(self, sigma):
        super(GaussianBlur_CUDA, self).__init__()
        self.sigma = sigma

    def forward(self, input):
        intBatches = input.size(0)
        intInputDepth = input.size(1)
        intInputHeight = input.size(2)
        intInputWidth = input.size(3)

        assert (input.is_contiguous() == True)

        output = input.new().resize_(intBatches, intInputDepth, intInputHeight, intInputWidth).zero_()
        temp = input.new().resize_(intBatches, intInputDepth, intInputHeight, intInputWidth).zero_()

        if input.is_cuda == True:
            DGPTCUDA.GaussianBlur_cuda_forward(
                input,
                temp,
                output,
                self.sigma
            )

        elif input.is_cuda == False:
            raise NotImplementedError()

        return output


if __name__ == '__main__':
    from PIL import Image
    import torchvision.transforms as T

    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")

    img = Image.open('/home/hao/Pictures/kitten.png')
    img = T.ToTensor()(img).unsqueeze(0).repeat(3,1,1,1).to(device)
    print(img.size())

    blur = GaussianBlur_CUDA()#.to(device)
    img = blur(img)
    # print(img)

    T.ToPILImage()(img[2].cpu()).save('blurkitten.png')

# end

# def backward(self, gradOutput):
# 	raise NotImplementedError() # BACKPROPAGATION NOT IMPLEMENTED
# end
# end
