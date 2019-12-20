from typing import Callable

import torch
import torch.nn as nn
import cv2
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from math import sin, cos, radians
import numpy as np
import collections
from itertools import repeat

class GaussianBlur(nn.Module):
    def __init__(self, kernel, sigma, nch):
        super(GaussianBlur, self).__init__()
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
            ww[i, i] = mk

        self.conv.weight = nn.Parameter(ww.data)
        self.conv.weight.requires_grad = False

    def forward(self, x):
        # print(self.conv)
        with torch.no_grad():
            o = self.conv(x)
        return o


def rotation_theta(degree, pos_x=0.0, pos_y=0.0):
    rad = radians(degree)
    return torch.Tensor([[[cos(rad), -sin(rad), pos_x], [sin(rad), cos(rad), pos_y]]])

def manipulate_tensor(input, rotation=0, scale=1, noise_strength=.0, shift_range=(.0, .0),
                      padding='zeros', gpu=True, size=0, offset_map =None, noise_blur_params = None,
                      output_size=None, align_corner=False):
    b, c, h, w = input.size()

    if not isinstance(scale, collections.Iterable):
        scale = tuple(repeat(scale, 2))

    if output_size is None:
        outh = int(scale[0] * h) if size == 0 else (size[0] if type(size) is tuple else size)
        outw = int(scale[1] * w) if size == 0 else (size[1] if type(size) is tuple else size)
    else:
        if not isinstance(output_size, collections.Iterable):
            output_size = tuple(repeat(output_size, 2))
        outh = output_size[0]
        outw = output_size[1]

    theta = rotation_theta(rotation).expand(b, 2, 3).to(input.device)

    grid = F.affine_grid(theta, torch.Size((b, c, outh, outw)))

    # if gpu:
    #     theta = theta.cuda()

    grid = grid.to(input.device) # unnecessary ?

    if noise_strength > 0:
        noise = torch.randn_like(grid)  * noise_strength
        if noise_blur_params is not None:
            blur = GaussianBlur(*noise_blur_params).to(input.device)
            noise = blur(noise.permute(0,3,1,2)).permute(0,2,3,1)
        grid += noise

    #if shift_range > 0:
        # print(theta.size())
    # shift_X = (np.random.random_sample() - 0.5) * shift_range
    # shift_Y = (np.random.random_sample() - 0.5) * shift_range

    grid[:, :, :, 1] += shift_range[1]
    grid[:, :, :, 0] += shift_range[0]

    if offset_map is not None:
        # print(grid.size())
        grid += offset_map

    if not align_corner:
        grid[:,:,:,0] *= (float(outh) - 1)*float(h) / (float(outh)*float(h-1))
        grid[:,:,:,1] *= (float(outw) - 1)*float(w) / (float(outw)*float(w-1))

    return F.grid_sample(input, grid.type_as(input), padding_mode=padding)

def tensor_center_crop(input, w, h):
    b, c, oh, ow = input.shape
    y = (oh - h) // 2
    x = (ow - w) // 2

    return input[:, :, y:y + h, x:x + w]

def tensor_shift(input, pixel_range=(1, -1), padding_mode='zeros'):
    b, c, h, w = input.shape
    theta = torch.Tensor([[
        [1, 0, 0], [0, 1, 0]
        ]]).to(input.device)
    grid = F.affine_grid(theta, torch.Size((b, c, h, w)))
    shift_h = pixel_range[1] / h
    shift_w = pixel_range[0] / w

    grid[:, :, :, 1] += shift_h
    grid[:, :, :, 0] += shift_w
    return F.grid_sample(input, grid, padding_mode=padding_mode)


def tensor_renorm(src, target):
    return src.sub(torch.mean(src))\
        .div(torch.std(src))\
        .mul(torch.std(target))\
        .add(torch.mean(target))

def rgb_renorm(src, target):
    assert src.size(1) == 3 and len(src.size()) == 4
    assert target.size(1) == 3 and len(target.size()) == 4

    target_mean = target.mean(2, keepdim=True).mean(3, keepdim=True)
    target_std = target.std(2, keepdim=True).std(3, keepdim=True)
    src_mean = src.mean(2, keepdim=True).mean(3, keepdim=True)
    src_std = src.std(2, keepdim=True).std(3, keepdim=True)

    return (src-src_mean)/src_std*target_std+target_mean


def batch_split_image(img_data, piece_size, padding=0):
    b, c, h, w = img_data.shape
    ps = piece_size
    ss = ps + padding * 2
    pe = padding
    px = (ps - w % ps) % ps
    py = (ps - h % ps) % ps

    pw = w // ps
    ph = h // ps

    if w % ps != 0:
        pw += 1

    if h % ps != 0:
        ph += 1

    p2d = (pe, px + pe, pe, py + pe)
    if pe != 0:
        img_data = F.pad(img_data, p2d)
        # print("PADDING to:", img_data.shape, pw, ph)

    ret = None

    if pe == 0:
        ret = img_data.view(b, c, ph, ps, pw, ps) \
            .permute(0, 1, 2, 4, 3, 5) \
            .permute(0, 2, 3, 1, 4, 5).contiguous() \
            .view(b * pw * ph, c, ps, ps)
    else:
        for i in range(ph):
            for j in range(pw):
                y = i * ps
                x = j * ps
                p = img_data[:, :, y:y + ss, x:x + ss]
                if ret is None:
                    ret = p
                else:
                    ret = torch.cat((ret, p), 0)

    return h, w, ph, pw, ret


def merge_images(img_out, piece_size, h, w, ph, pw, padding=0, scale=1):
    b, c, ps, ps = img_out.shape
    ret = None
    if padding != 0:
        ps = piece_size * scale
        pd = padding * scale
        for i in range(b):
            p = img_out[i:i + 1, :, pd:ps + pd, pd:ps + pd]
            if ret is None:
                ret = p
            else:
                ret = torch.cat((ret, p), 0)
    else:
        ret = img_out

    print(h, w, ph, pw, ps, c)

    ret = ret.view(1, ph, pw, c, ps, ps)
    ret = ret.permute(0, 3, 1, 2, 4, 5)
    ret = ret.permute(0, 1, 2, 4, 3, 5).contiguous()
    ret = ret.view(1, c, ph * ps, pw * ps)

    print("RET:", ret.shape)

    ret.squeeze_()
    ret = ret.cpu().detach()
    ret = ret[:, :h * scale, :w * scale]

    return ret

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


def flip(tensor, dims):
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

if __name__ == '__main__':
    print(torch.__version__)
    input = transforms.ToTensor()(Image.open('kitten.png'))
    c, h, w = input.size()
    # print(c,h,w)
    input = input.view(1, c, h, w)
    input = torch.cat((input, input), 0)

    input = manipulate_tensor(input, shift_range=0.1, gpu=False,
                              padding='border')  # , scale=1.3, noise_strength=0.1)
    # input = scale_tensor(input, 1.1)
    transforms.ToPILImage()(input[0, :, :, :].squeeze()).save('kitten_t.png')

to4d = lambda x:x.permute(0,2,1,3,4).contiguous().view(x.size(0)*x.size(2),x.size(1),x.size(3),x.size(4))
to5d = lambda x,d:x.view(-1, d, x.size(1), x.size(2), x.size(3)).permute(0,2,1,3,4).contiguous()