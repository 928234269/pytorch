import numpy as np
import torch
# from modules.stn import STN
from torchvision import transforms
from PIL import Image
from torch.autograd import Variable
from DGPT.Utils.TensorSTN import manipulate_tensor
import matplotlib.pyplot as plt
import cv2
import torch.nn as nn
from math import ceil

import PWCNet.PyTorch.models as models
from torchvision.models.resnet import resnet18

# define image dimensions

# class mask_transformer():

from .FlowMover import FlowMover

PWC_PATH = '/home/hao/PytorchModules/PWCNet/PyTorch/pwc_net.pth.tar'

class FloGenerator():
    def __init__(self, pwc_model_fn = PWC_PATH):
        self.net = self.load_pwcnet(pwc_model_fn)

    def load_pwcnet(self, pwc_model_fn):
        # pwc_model_fn = '/home/hao/PycharmProjects/FrameSynthesis/PWCNet/PyTorch/pwc_net.pth.tar'
        net = models.pwc_dc_net(pwc_model_fn)
        net = net.cuda()
        net.eval()
        net.requires_grad = False
        for p in net.parameters():
            p.requires_grad_(False)

        return net

    def get_flo_batch(self, first, second):
        b, c, H, W = first.shape
        # print(first.shape)

        divisor = 64
        # assert (H % divisor == 0)
        # assert (W % divisor == 0)

        H_ = int(ceil(H / divisor) * divisor)
        W_ = int(ceil(W / divisor) * divisor)

        im_all = torch.cat((first, second), 1)

        im_all = manipulate_tensor(im_all, size=(H_, W_))

        flo = self.net(im_all)
        flo = flo * 20.0

        # flo = flo.cpu().data.numpy()

        flo = manipulate_tensor(flo, size=(H, W))
        # print(flo.size())
        flo = flo.permute(0, 2, 3, 1).contiguous()
        flo[:, :, :, 0] *= H / float(H_)
        flo[:, :, :, 1] *= W / float(W_)
        return flo


class PreProcessor():
    def __init__(self, pwc_model_fn = PWC_PATH, extractor=None):
        self.flow_generator = FloGenerator(pwc_model_fn)
        self.flow_mover = ApplyFlow()

        if extractor is None:
            r18 = resnet18(pretrained=True)

            self.feature_extractor = r18.conv1
            self.feature_extractor.stride = 1
        else:
            self.feature_extractor = extractor


    def get_flow(self, input_a, input_b, dbg=1.0, t=1.0):
        a_to_b = self.flow_generator.get_flo_batch(input_a, input_b)
        return a_to_b


    def get_interpolated_of_extractor(self, input_a, input_b, input_A, input_B, extractor, comp=True, t=0.5):
        a_to_b = self.flow_generator.get_flo_batch(input_a, input_b)
        b_to_a = self.flow_generator.get_flo_batch(input_b, input_a)

        featout_a = None
        featout_b = None

        if extractor is not None:
            feat_a = extractor(input_A)
            feat_b = extractor(input_B)

            featout_a, wfa = self.flow_mover(feat_a, a_to_b * t)
            featout_b, wfb = self.flow_mover(feat_b, b_to_a * (1 - t))

            if comp:
                mfa = wfa.lt(0.5).float()
                mfb = wfb.lt(0.5).float()
                featout_a += mfa * featout_b
                featout_b += mfb * featout_a

        imgout_a, wa = self.flow_mover(input_a, a_to_b * t)
        imgout_b, wb = self.flow_mover(input_b, b_to_a * (1 - t))

        if comp:
            ma = wa.lt(0.5).float()
            mb = wb.lt(0.5).float()

            imgout_a += ma * imgout_b
            imgout_b += mb * imgout_a

        return featout_a, featout_b, imgout_a, imgout_b

if __name__== "__main__":
    from DGPT.Utils.VggFeatureExtrator import VggFeatureExtractor

    device = 'cuda'

    flow = PreProcessor(PWC_PATH, True)

    vgg = VggFeatureExtractor(conv_index=17, vgg=19, rgb_range=1).to(device)  # c 256
    vgg_half = VggFeatureExtractor(conv_index=8, vgg=19, rgb_range=1).to(device)  # c 128

    ff = ['left.png', 'mid.png', 'right.png']
    imgs = [Image.open(f) for f in ff]

    img_left, img_mid, img_right = imgs

    H = imgs[0].size[1]
    W = imgs[0].size[0]

    divisor = 64

    H_ = int(ceil(H / divisor) * divisor)
    W_ = int(ceil(W / divisor) * divisor)

    tf_resize = transforms.Resize((H_, W_))
    tf_half_resize = transforms.Resize((H_ // 2, W_ // 2))
    tf_small_resize = transforms.Resize((H_ // 4, W_ // 4))

    tf_tensor = transforms.Compose([
        transforms.ToTensor(),
        ])

    imgs = [tf_resize(x) for x in imgs]
    left_m, right_m, mid_m = [tf_tensor(tf_half_resize(x)) for x in imgs]
    left_s, right_s, mid_s = [tf_tensor(tf_small_resize(x)) for x in imgs]

    left, mid, right =[tf_tensor(x).unsqueeze(0).to(device) for x in imgs]

    with torch.no_grad():
        fa_q, fb_q, a_q, b_q = flow.get_interpolated_of_extractor(left_s, right_s, left, right, vgg)
        fa_h, fb_h, a_h, b_h = flow.get_interpolated_of_extractor(left_m, right_m, left, right,
                                                                  vgg_half)
        _, _, a_o, b_o = flow.get_interpolated_of_extractor(left, right, None, None, None)