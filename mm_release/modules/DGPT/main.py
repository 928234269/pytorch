import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import vgg19

import DGPT.DataLoader.Datasets.MixedDataset
import DGPT.DataLoader as DataLoader
from DGPT.DataLoader.DatasetLoader import load_dataset
from DGPT.Utils.Preprocess import YUV2RGB, RGB2YUV, GaussianBlur
from DGPT.Utils.AnalogPhoto import AnalogPhotoProcessor
from DGPT.Model.FeaturePolice import FeatureWorker, FeatureJudge
from DGPT.Visualize.Viz import Viz
from DGPT.Loss.PerceptualLoss import PerceptualLoss
from DGPT.Loss.GramMatrix import GramMatrix

import argparse

from PIL import ImageFile

#fix: large image file reading error
ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser(description='Train')
parser.add_argument('-lr', '--learning_rate', type=float, help='Learning rate', default=0.00001)
parser.add_argument('-e', '--epochs', type=int, help='training epochs', default=1000)
parser.add_argument('-b', '--batch_size', type=int, help='training data per batch', default=2)
parser.add_argument('-vs', '--valset_size', type=float, help='Validation set percentage per epoch', default=0.01)
parser.add_argument('-dim', '--input_dimension', type=int, help='size to resize input image to', default=448)

opt = parser.parse_args()

EPOCH = opt.epochs
DIM = opt.input_dimension
BATCH_SIZE = opt.batch_size
LR = opt.learning_rate

cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")
print('Using device:%s'%(device))

content_layers = ['conv_5']  # HAO['conv_4']
style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

vgg = vgg19(pretrained=True).to(device)
vgg.eval()
for p in vgg.parameters():
    p.requires_grad = False

def checkpoint_path(epoch, batch):
    if batch > 0:
        return '/media/hao/bigfatdrive/FeaturePolice/Checkpoints/worker_e_%d_b_%d' % (epoch, batch), '/media/hao/bigfatdrive/FeaturePolice/Checkpoints/judge_e_%d_b_%d' % (epoch, batch)
    else:
        return '/media/hao/bigfatdrive/FeaturePolice/Checkpoints/worker_e_%d' % (epoch), '/media/hao/bigfatdrive/FeaturePolice/Checkpoints/judge_e_%d' % (epoch)

def load_model(resume_e = 0, resume_b = 0):
    if resume_e > 0 or resume_b > 0:
        wp, jp = checkpoint_path(resume_e, resume_b)
        w = torch.load(wp)
        j = torch.load(jp)
    else:
        w = FeatureWorker()
        j = FeatureJudge(16,256)

    if not isinstance(w, nn.DataParallel):
        w = nn.DataParallel(w)
    if not isinstance(j, nn.DataParallel):
        j = nn.DataParallel(j)

    return w.to(device), j.to(device)

class PreProcess():
    def __init__(self, device):
        self.toRGB = YUV2RGB(device)
        self.toYUV = RGB2YUV(device)
        self.toAnalogPhoto = AnalogPhotoProcessor(0.05).to(device)
        self.baseline_blur = GaussianBlur(13, 3, 3).to(device)
        #self.baseline_blur = GaussianBlur(13,3,1).to(device)

    def proc(self, input):
        raw_yuv = self.toYUV.do(input)
        analog_rgb = self.toAnalogPhoto(input)
        input_yuv = self.toYUV.do(analog_rgb)
        baseline = self.baseline_blur(F.upsample(input_yuv, scale_factor=4, mode='bilinear'))

        #baseline = F.upsample(input_yuv, scale_factor=4, mode='bilinear')
        #baseline[:,:1,:,:] = self.baseline_blur(baseline[:,:1,:,:])
        return raw_yuv, input_yuv, baseline


def dump_worker_result(result):
    output = result.clone().detach().cpu()
    b, c, h, w = output.shape
    output = torch.reshape(output, (b * c, 1, h, w))
    for i in range(b*c):
        img = transforms.ToPILImage(mode='L')(output[i])
        img.save('./worker_result/y_%d.png' % i)


def train(resume_epoch=0, resume_batch=0):
    worker, judge = load_model(resume_epoch, resume_batch)
    processor = PreProcess(device)
    inner_criterion = nn.L1Loss()

    v = Viz()
    toRGB = YUV2RGB(device)

    worker_optimizer = torch.optim.Adam(worker.parameters(), lr = LR)
    judge_optimizer = torch.optim.Adam(judge.parameters(), lr=LR)

    batch_count = 0

    dataset = DataLoader.Datasets.MixedDataset.MixedDataset(
        root_dirs=[
            # dl.MixDatasetInfo('/home/le/Downloads/CelebA/Img/img_align_celeba_png/', 0, 0, 0),
            # dl.MixDatasetInfo('/media/hao/bigfatdrive/nn_pytorch_datasets/face_clip_551551.rgb', 1, 800, 1000),
            # dl.MixDatasetInfo('/media/hao/bigfatdrive/nn_pytorch_datasets/CelebA_HQ/datafiles/datafiles/', 2, 1024,
            #                   1024),
            DataLoader.Datasets.MixedDataset.MixDatasetInfo('/media/hao/bigfatdrive/face/101/', 0, 0, 0),
            DataLoader.Datasets.MixedDataset.MixDatasetInfo('/media/hao/bigfatdrive/face/01/', 0, 0, 0),
            DataLoader.Datasets.MixedDataset.MixDatasetInfo('/media/hao/bigfatdrive/face/02/', 0, 0, 0),
            DataLoader.Datasets.MixedDataset.MixDatasetInfo('/media/hao/bigfatdrive/face/03/', 0, 0, 0),

            # dl.MixDatasetInfo('/media/hao/bigfatdrive/nn_pytorch_datasets/Flickr2K/Flickr2K_HR/', 0, 0, 0),
        ],
        bs=DIM,
        testing=False,
    )

    perceptual_criterion = PerceptualLoss(vgg.features, inner_criterion, gram_slice=14)

    for e in range(EPOCH):
        train_loader, valid_loader = load_dataset(dataset, BATCH_SIZE, split=0.001, shuffle=True, random_seed=None)

        print('EPOCH %d, %d minibatches' % (e, len(train_loader)))
        # train model on training set
        for iter, data in enumerate(train_loader, 0):
            # print('minibatch %d'%(i))




            raw_images = data.to(device)
            raw_yuv, input_yuv, baseline_yuv = processor.proc(raw_images)

            raw_yuv = raw_yuv.detach()
            input_yuv = input_yuv.detach()
            baseline_yuv = baseline_yuv.detach()

            input_grayscale = input_yuv[:, :1, :, :]


            style_img_once = toRGB.do(raw_yuv).detach()

            for ii in range(1):
                judge_optimizer.zero_grad()
                worker_optimizer.zero_grad()
                worker_result = worker(input_grayscale)

                # if iter % 97 == 0:
                #     dump_worker_result(worker_result)


                judge_weight = judge(input_yuv)
                #F.softmax(judge_weight,1)
                #print(worker_result.mean())

                # upscales judge result to match dimensions of worker result
                judge_weight = F.upsample(judge_weight, scale_factor=4, mode='nearest')
                #judge_weight = F.upsample(judge_weight, scale_factor=4, mode='bilinear')

                # judge_noise = ((torch.rand_like(judge_weight)-0.5)*0.05).detach()
                judge_weight -= 0.0001

                # print(worker_result.size())
                # print(judge_weight.size())

                grayscale_sum = torch.sum(worker_result * judge_weight, 1)
                weight_sum = torch.sum(judge_weight, 1)
                diff_grayscale = grayscale_sum / weight_sum
                # print(diff_grayscale.size())
                b,c,h,w = baseline_yuv.size()
                result_y = baseline_yuv[:,:1,:,:] + diff_grayscale.view(b,1,h,w)
                result = torch.cat((result_y, baseline_yuv[:,1:,:,:]), 1)

                # loss = criterion(result[:,:1,:,:], raw_yuv[:,:1,:,:])



                # TODO: refactor style loss
                input_img = toRGB.do(result)
                style_img = style_img_once.clone()

                loss = perceptual_criterion(input_img, style_img)
                loss.backward()

                judge_optimizer.step()
                worker_optimizer.step()

            v.draw_line(batch_count, loss.item(), 'L1_LOSS')
            batch_count += 1

            if iter % 11 == 0:
                v.draw_images(result[:,:1,:,:], 'RESULT')
                v.draw_images(raw_yuv[:,:1,:,:], 'LABEL')
                v.draw_images(input_yuv[:,:1,:,:], 'INPUT')
                #v.draw_images(toRGB.do(result), 'RESULT')
                #v.draw_images(toRGB.do(raw_yuv), 'LABEL')
                #v.draw_images(toRGB.do(input_yuv), 'INPUT')
                #v.draw_images(result[:,:1,:,:]-raw_yuv[:,:1,:,:], 'DIFF_GT')
                v.draw_images(diff_grayscale.view(b,1,h,w), 'DIFF_OUTPUT')
                v.draw_images(baseline_yuv[:,:1,:,:], 'BASELINE')
                #v.draw_images(toRGB.do(baseline_yuv), 'BASELINE')

            if iter % 500 == 0 and iter > 0:
                wp, jp = checkpoint_path(e, iter)
                torch.save(worker, wp)
                torch.save(judge, jp)

        wp, jp = checkpoint_path(e, 0)
        torch.save(worker, wp)
        torch.save(judge, jp)
        #TODO: validate model on validation set




if __name__ == "__main__":
    train(0,0)








