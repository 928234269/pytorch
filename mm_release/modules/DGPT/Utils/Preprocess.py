import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import torchvision.transforms as T

from math import acos, sqrt

from DGPT.Utils.TensorSTN import manipulate_tensor, tensor_center_crop
from DGPT.Utils.CUDAFuncs.GaussianBlur import GaussianBlur_CUDA as CudaBlur

import io
from PIL import Image



class YUV2RGB():
    def __init__(self, device=None):
        self.conv_matrix = torch.zeros([3, 3, 1, 1], dtype=torch.float)

        self.conv_matrix[0][0][0][0] = 1
        self.conv_matrix[0][1][0][0] = 0
        self.conv_matrix[0][2][0][0] = 1.13983
        self.conv_matrix[1][0][0][0] = 1
        self.conv_matrix[1][1][0][0] = -0.39465
        self.conv_matrix[1][2][0][0] = -0.58060
        self.conv_matrix[2][0][0][0] = 1
        self.conv_matrix[2][1][0][0] = 2.03211
        self.conv_matrix[2][2][0][0] = 0
        # self.bias = torch.Tensor([-0.7, 0.528, -0.683])
        # self.bias = torch.Tensor([0.0, 0.0, 0.0]).to(device)

        '''
        self.conv_matrix[0][0][0][0] = 1
        self.conv_matrix[0][1][0][0] = 0
        self.conv_matrix[0][2][0][0] = 0
        self.conv_matrix[1][0][0][0] = 1
        self.conv_matrix[1][1][0][0] = 0
        self.conv_matrix[1][2][0][0] = 0
        self.conv_matrix[2][0][0][0] = 1
        self.conv_matrix[2][1][0][0] = 0
        self.conv_matrix[2][2][0][0] = 0
        # self.bias = torch.Tensor([-0.7, 0.528, -0.683])
        self.bias = torch.Tensor([0.0, 0.0, 0.0]).to(device)
        '''
        # self.conv_matrix = self.conv_matrix.to(device)


    def do(self, input):
        c = self.conv_matrix.to(input.device)
        rgb = F.conv2d(input, c.view(3, 3, 1, 1), stride=1,
                       padding=0)

        return rgb


class YUV2BGR():
    def __init__(self, device):
        self.toRGB = YUV2RGB(device)

    def do(self, input):
        rgb = self.toRGB.do(input)

        return rgb[:, [2, 1, 0], :, :]


class RGB2YUV():
    '''
       outputY:zero():add(0.299, inputRed):add(0.587, inputGreen):add(0.114, inputBlue)
   outputU:zero():add(-0.14713, inputRed):add(-0.28886, inputGreen):add(0.436, inputBlue)
outputV:zero():add(0.615, inputRed):add(-0.51499, inputGreen):add(-0.10001, inputBlue)
    '''
    def __init__(self, device=None):
        self.conv_matrix = torch.zeros([3, 3, 1, 1], dtype=torch.float)

        self.conv_matrix[0][0][0][0] = 0.299
        self.conv_matrix[0][1][0][0] = 0.587
        self.conv_matrix[0][2][0][0] = 0.114
        self.conv_matrix[1][0][0][0] = -0.14713
        self.conv_matrix[1][1][0][0] = -0.28886
        self.conv_matrix[1][2][0][0] = 0.436 #HAO!!!!
        self.conv_matrix[2][0][0][0] = 0.615
        self.conv_matrix[2][1][0][0] = -0.51499
        self.conv_matrix[2][2][0][0] = -0.10001
        # self.bias = torch.Tensor([-0.7, 0.528, -0.683])
        # self.bias = torch.Tensor([0.0, 0.0, 0.0]).to(device)
        # self.conv_matrix = self.conv_matrix.to(device)


    def do(self, input):
        c = self.conv_matrix.to(input.device)
        yuv = F.conv2d(input, c.view(3, 3, 1, 1), bias=None, stride=1,
                       padding=0)
        return yuv

class RGB2Y():
    '''
       outputY:zero():add(0.299, inputRed):add(0.587, inputGreen):add(0.114, inputBlue)
   outputU:zero():add(-0.14713, inputRed):add(-0.28886, inputGreen):add(0.436, inputBlue)
outputV:zero():add(0.615, inputRed):add(-0.51499, inputGreen):add(-0.10001, inputBlue)
    '''
    def __init__(self, device=None):
        self.conv_matrix = torch.zeros([1, 3, 1, 1], dtype=torch.float)

        self.conv_matrix[0][0][0][0] = 0.299
        self.conv_matrix[0][1][0][0] = 0.587
        self.conv_matrix[0][2][0][0] = 0.114


    def do(self, input):
        c = self.conv_matrix.to(input.device)
        y = F.conv2d(input, c.view(1, 3, 1, 1), bias=None, stride=1,
                       padding=0)
        return y

class RGB2OldYYY():
    '''
       outputY:zero():add(0.299, inputRed):add(0.587, inputGreen):add(0.114, inputBlue)
   outputU:zero():add(-0.14713, inputRed):add(-0.28886, inputGreen):add(0.436, inputBlue)
outputV:zero():add(0.615, inputRed):add(-0.51499, inputGreen):add(-0.10001, inputBlue)
    '''
    def __init__(self,  device=None):
        self.conv_matrix = torch.zeros([3, 3, 1, 1], dtype=torch.float)

        self.conv_matrix[0][0][0][0] = 0.05
        self.conv_matrix[0][1][0][0] = 0.40
        self.conv_matrix[0][2][0][0] = 0.55

        self.conv_matrix[1][0][0][0] = 0.05
        self.conv_matrix[1][1][0][0] = 0.40
        self.conv_matrix[1][2][0][0] = 0.55

        self.conv_matrix[2][0][0][0] = 0.05
        self.conv_matrix[2][1][0][0] = 0.40
        self.conv_matrix[2][2][0][0] = 0.55


    def do(self, input):
        c = self.conv_matrix.to(input.device)
        y = F.conv2d(input, c.view(3, 3, 1, 1), bias=None, stride=1,
                       padding=0)
        y = ((y - 0.5) * 4).sigmoid()


        return y


class RGB2HY():
    '''
       outputY:zero():add(0.299, inputRed):add(0.587, inputGreen):add(0.114, inputBlue)
   outputU:zero():add(-0.14713, inputRed):add(-0.28886, inputGreen):add(0.436, inputBlue)
outputV:zero():add(0.615, inputRed):add(-0.51499, inputGreen):add(-0.10001, inputBlue)
    '''
    def __init__(self, device):
        self.y_conv_matrix = torch.zeros([1, 3, 1, 1], dtype=torch.float)

        self.y_conv_matrix[0][0][0][0] = 0.299
        self.y_conv_matrix[0][1][0][0] = 0.587
        self.y_conv_matrix[0][2][0][0] = 0.114
        # self.conv_matrix[1][0][0][0] = -0.14713
        # self.conv_matrix[1][1][0][0] = -0.28886
        # self.conv_matrix[1][2][0][0] = 0.436 #HAO!!!!
        # self.conv_matrix[2][0][0][0] = 0.615
        # self.conv_matrix[2][1][0][0] = -0.51499
        # self.conv_matrix[2][2][0][0] = -0.10001
        # self.bias = torch.Tensor([-0.7, 0.528, -0.683])
        self.y_bias = torch.Tensor([0.0]).to(device)
        self.y_conv_matrix = self.y_conv_matrix.to(device)


    def do(self, input):
        batch,c,h,w = input.size()
        y = F.conv2d(input, self.y_conv_matrix.view(1, 3, 1, 1), bias=self.y_bias, stride=1,
                       padding=0)
        r = input[:,0,:,:]
        g = input[:,1,:,:]
        b = input[:,2,:,:]

        r_g = r - g
        r_b = r - b
        g_b = g - b

        hue = ( ( (r_g + r_b) / (2*torch.sqrt(r_g*r_g+r_b*g_b)) ) + 1 ) / 2
        hue = hue.view(batch, 1, h, w)
        return torch.cat((hue,y), 1)

# class HY2RGB():
#     def __init__(self):

class RGB2XYZ():
    def __init__(self, device):
        self.conv_matrix = torch.zeros([3, 3, 1, 1], dtype=torch.float)

        self.conv_matrix[0][0][0][0] = 0.412453
        self.conv_matrix[0][1][0][0] = 0.357580
        self.conv_matrix[0][2][0][0] = 0.180423
        self.conv_matrix[1][0][0][0] = 0.212671
        self.conv_matrix[1][1][0][0] = 0.715160
        self.conv_matrix[1][2][0][0] = 0.072169
        self.conv_matrix[2][0][0][0] = 0.019334
        self.conv_matrix[2][1][0][0] = 0.119193
        self.conv_matrix[2][2][0][0] = 0.950227

        self.bias = torch.Tensor([0.0, 0.0, 0.0]).to(device)
        self.conv_matrix = self.conv_matrix.to(device)

    def do(self, input):
        xyz = F.conv2d(input, self.conv_matrix.view(3, 3, 1, 1), bias=self.bias, stride=1,
                       padding=0)
        return xyz

class XYZ2RGB():
    def __init__(self, device):
        self.conv_matrix = torch.zeros([3, 3, 1, 1], dtype=torch.float)

        self.conv_matrix[0][0][0][0] = 3.240479
        self.conv_matrix[0][1][0][0] = -1.537150
        self.conv_matrix[0][2][0][0] = -0.498535
        self.conv_matrix[1][0][0][0] = -0.969256
        self.conv_matrix[1][1][0][0] = 1.875992
        self.conv_matrix[1][2][0][0] = 0.041556
        self.conv_matrix[2][0][0][0] = 0.055648
        self.conv_matrix[2][1][0][0] = -0.204043
        self.conv_matrix[2][2][0][0] = 1.057311

        self.bias = torch.Tensor([0.0, 0.0, 0.0]).to(device)
        self.conv_matrix = self.conv_matrix.to(device)

    def do(self, input):
        rgb = F.conv2d(input, self.conv_matrix.view(3, 3, 1, 1), bias=self.bias, stride=1,
                       padding=0)
        return rgb

class XYZ2LAB():
    def __init__(self, device):
        self.XN = 0.9515
        self.YN = 1.0000
        self.ZN = 1.0886

    def f(self, input_tensor):
        cond = input_tensor.gt(0.008856).float()
        return input_tensor**0.333333*cond + (1-cond)*7.787*input_tensor+0.137931

    def do(self, input):
        b,c,h,w = input.size()
        x = input[:,0,:,:]/self.XN
        y = input[:, 1, :, :]
        z = input[:, 2, :, :] / self.XN

        l_condition = (y > 0.008856).float()

        l = ((116*(y**0.3333)-16) * l_condition + (1 - l_condition) * 903.3 * y).view(b,1,h,w)
        a = (500* (self.f(x) - self.f(y))).view(b,1,h,w)
        b = (200* (self.f(y) - self.f(z))).view(b,1,h,w)

        return torch.cat((l,a,b), 1)

class LAB2XYZ():
    def __init__(self, device):
        self.XN = 0.9515
        self.YN = 1.0000
        self.ZN = 1.0886

    def f(self, input_tensor, n_value):
        cond = input_tensor.gt(0.008856).float()
        return n_value * input_tensor ** 3 * cond + (1-cond)*(input_tensor - 16)/116*3*0.008856*0.008856*n_value

    def do(self, input):
        batch, c, h, w = input.shape
        l = input[:,0,:,:]
        a = input[:, 1, :, :]
        b = input[:, 2, :, :]

        fy = (l + 16) / 116
        fx = fy + a / 500
        fz = fy - b / 200


        y = self.f(fy, self.YN).view(batch,1,h,w)
        x = self.f(fx, self.XN).view(batch,1,h,w)
        z = self.f(fz, self.ZN).view(batch,1,h,w)

        return torch.cat((x,y,z), 1)

class RGB2LAB():
    def __init__(self, device):
        self.toxyz = RGB2XYZ(device)
        self.tolab = XYZ2LAB(device)

    def do(self, input):
        return self.tolab.do(self.toxyz.do(input))


class LAB2RGB():
    def __init__(self, device):
        self.toxyz = LAB2XYZ(device)
        self.torgb = XYZ2RGB(device)

    def do(self, input):
        return self.torgb.do(self.toxyz.do(input))

class MotionBlurX(nn.Module):
    def __init__(self, kernel, sigma, nch):
        super(MotionBlurX, self).__init__()
        padding = kernel // 2
        self.conv = nn.Conv2d(nch, nch, (1,kernel), stride=1, padding=(0,padding), bias=False)
        gk = cv2.getGaussianKernel(kernel, sigma)
        # wk = torch.eye(kernel) * torch.Tensor(gk)
        k = torch.Tensor(gk)
        tk = k.t()
        mk = k * tk

        # self.conv.weight\
        tmp_weight = nn.Parameter(torch.zeros(nch, nch, kernel, kernel))
        # ww = self.conv.weight.clone()

        for i in range(nch):
            tmp_weight[i, i] = mk

        self.conv.weight = nn.Parameter(tmp_weight.data.sum(2, keepdim=True))
        self.conv.weight.requires_grad = False

    def forward(self, x):
        # print(self.conv)
        o = self.conv(x)
        return o

class GaussianConvBlur(nn.Module):
    def __init__(self, kernel, sigma, nch):
        super(GaussianConvBlur, self).__init__()
        self.padding = kernel // 2
        self.conv = nn.Conv2d(nch, nch, kernel, stride=1, padding=self.padding, bias=False)
        gk = cv2.getGaussianKernel(kernel, sigma)
        # wk = torch.eye(kernel) * torch.Tensor(gk)
        k = torch.Tensor(gk)
        tk = k.t()
        mk = k * tk

        self.conv.weight = nn.Parameter(torch.zeros(nch, nch, kernel, kernel))
        ww = self.conv.weight.clone()

        for i in range(nch):
            ww[i, i] = mk

        self.conv.weight = nn.Parameter(ww.data)
        self.conv.weight.requires_grad = False
        self.nch = nch
    def forward(self, x, noise=0):
        weight = self.conv.weight.clone()
        if noise>0:
            n = noise*torch.randn_like(self.conv.weight[0,0])
            n = n-n.mean()
            for i in range(self.nch):
                #nn=n[i,i]-n[i,i].mean()
                weight[i,i]+=n

        o = F.conv2d(x,  weight, padding=self.padding)
        return o

_topil = T.ToPILImage()
_totensor = T.ToTensor()

def JPEG_Compression(tensor, quality_min, quality_max):
    batch = tensor.size(0)
    out = torch.zeros_like(tensor)
    for i in range(batch):
        p = _topil(tensor[i].cpu())
        buffer = io.BytesIO()
        p.save(buffer, "JPEG", quality=np.random.randint(quality_min, quality_max))
        p = Image.open(buffer)
        t = _totensor(p)
        out[i] = t

    return out


class GaussianBlur(nn.Module):
    def __init__(self, kernel=3, sigma=3, nch=3, dilation=1):
        super(GaussianBlur, self).__init__()
        self.gaussian = CudaBlur(sigma)

    def forward(self, x):
        with torch.no_grad():
            o = self.gaussian(x)
        return o

def SpatialSubNormFunc(kernel, sigma, nch, device='cuda'):
    def _ssn(input):
        output = F.conv2d(input, weight, padding=padding)
        return output

    padding = kernel // 2
    gk = cv2.getGaussianKernel(kernel, sigma)
    k = torch.Tensor(gk)
    tk = k.t()
    mk = k * tk

    ww = nn.Parameter(torch.zeros(nch, nch, kernel, kernel))

    for i in range(nch):
        ww[i, i, padding, padding] = 1
        ww[i, i] -= mk
    weight = ww.data

    if torch.cuda.is_available():
        weight = weight.to(device)

    return _ssn


def MidPassFilterFunc(kernel=9, sigma0=2.0,sigma1=2.1, threshold=0.0004, nch=1, device='cuda'):
    def _mpf(input):
        output = F.conv2d(input, weight, padding=padding)
        output -= th
        output = F.relu(output)
        return output

    padding = kernel // 2
    gk0 = cv2.getGaussianKernel(kernel, sigma0)
    k = torch.Tensor(gk0)
    tk = k.t()
    mk0 = k * tk
    gk1 = cv2.getGaussianKernel(kernel, sigma1)
    k = torch.Tensor(gk1)
    tk = k.t()
    mk1 = k * tk

    ww = nn.Parameter(torch.zeros(nch, nch, kernel, kernel))

    for i in range(nch):
        ww[i, i] = mk0 - mk1
    weight = ww.data

    weight = weight.to(device)

    th = threshold

    return _mpf
    
def UnsharpMaskFunc(kernel, sigma, percent, threshold, nch):
    def _usm(input):
        output = F.conv2d(input, weight, padding=padding)

        output[output.gt(-threshold) * output.lt(threshold)] = 0

        output = input + output * percent / 100
        return output.clamp(0, 1)

    threshold /= 255
    padding = kernel // 2
    gk = cv2.getGaussianKernel(kernel, sigma)
    k = torch.Tensor(gk)
    tk = k.t()
    mk = k * tk

    ww = nn.Parameter(torch.zeros(nch, nch, kernel, kernel))

    for i in range(nch):
        ww[i, i, padding, padding] = 1
        ww[i, i] -= mk
    weight = ww.data

    if torch.cuda.is_available():
        weight = weight.to('cuda')

    return _usm


def AsymBlurFunc(kernel_h, sigma_h, kernel_v, sigma_v, nch, device):
    def _asb(input):
        output = F.conv2d(input, weight, stride=1, padding=padding)
        return output

    padding = [kernel_v // 2, kernel_h // 2]

    gk_v = cv2.getGaussianKernel(kernel_v, sigma_v)
    k_v = torch.Tensor(gk_v)

    gk_h = cv2.getGaussianKernel(kernel_h, sigma_h)
    k_h = torch.Tensor(gk_h)

    tk_h = k_h.t()
    mk = k_v * tk_h

    ww = nn.Parameter(torch.zeros(nch, nch, kernel_v, kernel_h))

    for i in range(nch):
        ww[i, i] = mk

    weight = ww.data

    weight = weight.to(device)

    return _asb


def NaiveDIFunc(nch):
    def _di(input):
        input = F.pad(input, (0, 0, 1, 1), 'replicate')
        output = F.conv2d(input, weight, stride=1, padding=(0, 0))
        #        print(output.shape)
        #        output = output.reshape(output.shape[0]*2,nch, 2*output.shape[2],output.shape[3])
        w = output.shape[3]
        h = output.shape[2]
        b = output.shape[0]
        output = output.reshape(b, nch, 2, 2, h, w)
        output = output.permute(0, 2, 1, 4, 3, 5).reshape(b * 2, nch, h * 2, w)
        output = output.narrow(2, 1, (h - 1) * 2).contiguous()
        return output

    ww = nn.Parameter(torch.zeros(nch * 4, nch * 2, 2, 1))

    for i in range(nch):
        # even
        ww[i * 4, i + 0, 0, 0] = 0.5
        ww[i * 4, i + 0, 1, 0] = 0.5
        ww[i * 4 + 1, i + 0, 1, 0] = 1
        # odd
        ww[i * 4 + 2, i + nch, 0, 0] = 1
        ww[i * 4 + 2 + 1, i + nch, 0, 0] = 0.5
        ww[i * 4 + 2 + 1, i + nch, 1, 0] = 0.5

    weight = ww.data

    if torch.cuda.is_available():
        weight = weight.to('cuda')

    return _di


class SpatialSubNorm(nn.Module):
    def __init__(self, kernel, sigma, nch):
        super(SpatialSubNorm, self).__init__()
        padding = kernel // 2
        self.conv = nn.Conv2d(nch, nch, kernel, stride=1, padding=padding, bias=False)
        gk = cv2.getGaussianKernel(kernel, sigma)
        # wk = torch.eye(kernel) * torch.Tensor(gk)
        k = torch.Tensor(gk)
        tk = k.t()
        mk = k * tk



        self.conv.weight = nn.Parameter(torch.zeros(nch, nch, kernel, kernel))
        ww = self.conv.weight.clone()


        for i in range(nch):
            ww[i,i,padding,padding]=1
            ww[i, i] -= mk

        self.conv.weight = nn.Parameter(ww.data)


    def forward(self, x):
        # print(self.conv)
        # with torch.no_grad():
        self.conv.weight.requires_grad = False
        o = self.conv(x)
        return o


# TODO: additional noise info based on current noise frequency (post process)


def multi_meshgrid(*args):
    """
    Creates a meshgrid from possibly many
    elements (instead of only 2).
    Returns a nd tensor with as many dimensions
    as there are arguments
    """
    args = list(args)
    template = [1 for _ in args]
    for i in range(len(args)):
        n = args[i].shape[0]
        template_copy = template.copy()
        template_copy[i] = n
        args[i] = args[i].view(*template_copy)
        # there will be some broadcast magic going on
    return tuple(args)


def tensor_flip(tensor, dims):
    if not isinstance(dims, (tuple, list)):
        dims = [dims]
    indices = [torch.arange(tensor.shape[dim] - 1, -1, -1,
        dtype=torch.int64) for dim in dims]
    multi_indices = multi_meshgrid(*indices)
    final_indices = [slice(i) for i in tensor.shape]
    for i, dim in enumerate(dims):
        final_indices[dim] = multi_indices[i]
    flipped = tensor[final_indices]
    # need to permute the final dimensions
    # if dims is not consecutive, but I'm lazy
    # now :-)
    return flipped

def tensor_gc_norm(src):
    mean = torch.tensor([torch.mean(src[:,0]).item(), torch.mean(src[:,1]).item(), torch.mean(src[:,2]).item()]).to(src.device)
    std = torch.tensor([torch.std(src[:,0]).item(), torch.std(src[:,1]).item(), torch.std(src[:,2]).item()]).to(src.device)
    return src.sub(mean.view(1,3,1,1))\
        .div(std.view(1,3,1,1)+0.00000001)

def tensor_normalize(batch, from_255=False):
    # normalize using imagenet mean and std
    dim = len(batch.size())
    if dim == 3:
        mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    elif dim == 4:
        mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 3, 1, 1)
        std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 3, 1, 1)
    else:
        mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        
    if from_255:
        batch = batch.div_(255.0)
    return batch.sub(mean).div(std)


def tensor_unnormalize(batch, from_255=False, to_255=False):
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)

    if from_255:
        batch = batch.div_(255.0)

    batch = batch.mul(std).add(mean)

    if to_255:
        batch = batch.mul(255.0)
    return batch


def tensor_normalize_simple(batch):
    return batch.sub(0.45).div(0.226)


def tensor_unnormalize_simple(batch):
    batch = batch.mul(0.226).add(0.45)

    return batch


def recover_image(img):
    return (
        (
            img *
            np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1)) +
            np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))
        ).transpose(0, 2, 3, 1) * 255.).clip(0, 255).astype(np.uint8)


def tensor_evil_unnormalize(batch):
    b, c, h, w = batch.shape
    mean = batch.mean(3).mean(2).view(b, c, 1, 1)
    std = batch.std(3).std(2).view(b, c, 1, 1)

    return batch.mul(std).add(mean)


def generate_real_image_pair(input_tensor, rotation=10, shift_range=(.0, .0), noise_strength = 0.01, grid_out_size = 11):

    a = manipulate_tensor(input_tensor,
                          rotation=rotation,
                          shift_range=shift_range,
                          noise_strength=noise_strength
                          )
    a = tensor_center_crop(a, grid_out_size, grid_out_size)
    b = manipulate_tensor(input_tensor,
                          rotation=-rotation,
                          shift_range=[-x for x in shift_range],
                          noise_strength=noise_strength)
    b = tensor_center_crop(b, grid_out_size, grid_out_size)
    return a, b
