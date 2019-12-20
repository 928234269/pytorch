import json

import torch
import torch.nn.functional as F
import torchvision.transforms as T

from nn_pytorch.face2c.helper import TaskProcessor, ServerInfo, CUDNNRequestHandler, InferenceError
#from DGPT.Utils.FaceUtils import FaceCutter, rotateByEXIF

from PIL import Image

import os
from os import listdir
from os.path import join, basename, splitext

def svc_get_tensor_result(tensor, upscale, type):
    print("send tensor request", type, tensor.dtype)
    output, = opt.cudnn_svc.run(tensor.half(), type)

    print(output.device, output.shape)
    return 0, output.float()

def process_full(task, input, method='gpu', bg_pool=1.0):
    faces_result = None
    tf = T.ToTensor()

    input_data = tf(input).unsqueeze(0).to('cuda')

    diagnose_input(opt.rgb2yuv, task, input_data)

    # process faces if needed
    if len(task.faces) > 0 and not task.is_eye:
        faces_data = []
        faces_base = []
        for i, f in zip(range(len(task.faces)), task.faces):
            if opt.testing:
                f.save(f"./tests/face_orig_{i}.png")

            face = tf(f).unsqueeze(0).to('cuda')
            face, base_blur = preprocess_face(face, task.upscale)
            faces_data.append(face)
            faces_base.append(base_blur)

        faces_input = torch.cat(faces_data, 0)
        faces_base_input = torch.cat(faces_base, 0)

        faces_result = process_faces(faces_input, faces_base_input, False)

        if opt.testing:
            for i in range(len(faces_result)):
                print(faces_result[i].size)
                faces_result[i].save(f"./tests/face_result_{i}.png")
                task.faces[i].save(f"./tests/face_input_{i}.png")
                T.ToPILImage()(faces_base_input[i].clamp(0, 1).cpu()).save(f"./tests/face_base_{i}.png")
                T.ToPILImage()(faces_data[i].squeeze(0).clamp(0, 1).cpu()).save(f"./tests/face_preprocessed_{i}.png")


        del faces_input
        del faces_base_input

    # process full img

    # input_data = opt.bg_valid_blur(input_data)
    # input_data = prepare_data(input_data)
    # input_data = perform_blur(opt.bg_valid_blur, input_data)

    b, c, h, w = input_data.shape
    bg_scale = task.upscale

    input_data, bg_upblur = preprocess_bg(input_data, bg_scale, bg_pool)
    if opt.testing:
        T.ToPILImage()(bg_upblur.squeeze(0).clamp(0, 1).cpu()).save("./tests/bg_base.png")

    ph0, ph1, pw0, pw1 = [0] * 4
    min_sz = 256
    if h < min_sz or w < min_sz:

        ph0 = 0 if h > min_sz else (min_sz - h) // 2
        pw0 = 0 if w > min_sz else (min_sz - w) // 2

        ph1 = ph0
        pw1 = pw0
        if h + ph0 + ph1 < min_sz:
            ph1 += min_sz - h - ph0 - ph1

        if w + pw0 + pw1 < min_sz:
            pw1 += min_sz - w - pw0 - pw1

        input_data = F.pad(input_data, (pw0, pw1, ph0, ph1))

    if method == 'cpu':
        code, result = svc_get_result(input_data, bg_scale, 2)
    else:
        code, result = svc_get_tensor_result(input_data, bg_scale, 2)

    print("BG result shape", code, result.shape, result.device)

    if code != 0:
        raise InferenceError(code,
                             "cudnn backend wrong")

    # print("result size", input_data.shape, result.shape, [x.size for x in faces_result])
    if result.shape[2] > h * bg_scale or result.shape[3] > w * bg_scale:
        print("FULL output", result.shape, " crop ",
              ph0 * bg_scale, "to", -ph1 * bg_scale,
              "w", pw0 * bg_scale, "to", -pw1 * bg_scale)
        top = ph0 * bg_scale
        bottom = -ph1 * bg_scale
        left = pw0 * bg_scale
        right = -pw1 * bg_scale
        if bottom == 0:
            bottom = ph0 + h * bg_scale
        if right == 0:
            right = pw0 + w * bg_scale

        result = result[:, :, top:bottom, left:right].contiguous()

    if opt.config.bg_net.use_base:
        print(result.type(), bg_upblur.type())
        if method == 'cpu':
            result = result.float() + bg_upblur.cpu()
        else:
            result = result.float() + bg_upblur


    # if task.is_gray:
    #     noise = torch.randn_like(result) * opt.config.grayscale_noise_strength
    #     result += noise
    #     # result.clamp_(0, 1)
    return result, faces_result


def preprocess_bg(bg, scale, pool=1):
    # photo = opt.bg_valid_blur(bg)
    photo = perform_blur(opt.bg_valid_blur, bg)

    if pool > 1:
        # bg = F.avg_pool2d(bg, pool, pool)
        photo = F.interpolate(photo, scale_factor=1 / pool, mode='bilinear', align_corners=False)

    upblurphoto = None
    if opt.config.bg_net.use_base:
        upblurphoto = photo.clone()
        upblurphoto = manipulate_tensor(upblurphoto, scale=scale)
        # upblurphoto = opt.bg_base_blur(upblurphoto)
        upblurphoto = perform_blur(opt.bg_base_blur, upblurphoto)


    return photo, upblurphoto


import sys
import yaml
import argparse
from nn_pytorch.OnePage.engine_utils import dict2obj


def parse_cmd():
    parser = argparse.ArgumentParser(description='service')
    parser.add_argument("--config", type=str, default='./config.yaml', help="config file")
    parser.add_argument("--testing", type=bool, default=False, help="testing flag")
    parser.add_argument('--fp16', help='', action='store_true', default=False)
    parser.add_argument('--mgpu', help='', action='store_true', default=False)

    opt = parser.parse_args()
    return opt


def load_config(conf_file):
    with open(conf_file, 'r') as stream:
        try:
            opt_data = yaml.full_load(stream)
        except yaml.YAMLError as exc:
            print("options config format error", exc)
            sys.exit(1)

    options = dict2obj(opt_data)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    options.device = device

    return options

opt = None

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG', '.tif'])

if __name__ == "__main__":
    print("start")

    args = parse_cmd()
    opt = load_config(args.config)
    opt.testing = True
    opt.fp16 = True
    opt.args = args

    # opt.face_cutter = FaceCutter(512,
    #                              cuda=True,
    #                              blur_threshold=opt.config.dface.blur_threshold,
    #                              method=opt.config.dface.method,
    #                              mtcnn_path=opt.config.dface.mtcnn_path,
    #                              small_face_size=opt.config.dface.small_face_size,
    #                              tiny_face_size=opt.config.dface.tiny_face_size,
    #                              mtcnn_thresholds=opt.config.dface.mtcnn_thresholds,
    #                              mtcnn_nms_thresholds=opt.config.dface.mtcnn_nms_thresholds
    #                              )

    from DGPT.Utils.CUDAFuncs.GaussianBlur import GaussianBlur_CUDA

    # opt.face_initial_blur = GaussianBlur_CUDA(opt.face_net.init_sigma)
    print("base sigma=",opt.config.face_net.base_sigma)
    opt.face_base_blur = GaussianBlur_CUDA(opt.config.face_net.base_sigma)
    opt.face_valid_blur = GaussianBlur_CUDA(opt.config.face_net.valid_sigma)

    opt.bg_valid_blur = GaussianBlur_CUDA(opt.config.bg_net.valid_sigma)
    opt.bg_base_blur = GaussianBlur_CUDA(opt.config.bg_net.base_sigma)

    # from quicklab.utils import VggUnNormalizer
    # opt.vgg_unnorm = VggUnNormalizer().to('cuda')

    from DGPT.Utils.Preprocess import RGB2YUV, YUV2RGB
    opt.rgb2yuv = RGB2YUV('cuda')
    opt.yuv2rgb = YUV2RGB('cuda')

    # prepare mask
    face_mask = torch.ones(1, 1, 512, 512).to('cuda')
    face_mask *= 0.1
    face_mask[:, :, 50:462, 144:512 - 144] = 1.35
    face_mask = opt.face_base_blur(face_mask)
    face_mask = opt.face_base_blur(face_mask)

    opt.output_mask = F.relu(face_mask - 0.8) + 0.8


    opt.fingerprint = None
    # opt.cudnn_client = CUDNNRequestHandler()
    import mmcudnnsvc
    # opt.cudnn_svc = cudnnsvc.CudnnSvc(4, "/workspace/release_0423/face2c/models", "Ax2.bin", "JK.bin", True)
    opt.cudnn_svc = mmcudnnsvc.CudnnSvc(2, "/workspace/mm_release/face2c_cudnn/models", "MM.bin", "MM.bin", True)

    test_dir = '/workspace/test_dir'
    filenames = [join(test_dir, x) for x in listdir(test_dir) if is_image_file(x)]
    filenames.sort()
    output_dir = join(test_dir, "outputs")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    bg_scale = 2
    for fn in filenames:
        img = Image.open(fn).convert("RGB")
        input_data = T.ToTensor()(img).unsqueeze(0).cuda()
        input_data = opt.bg_valid_blur(input_data)
        bg_upblur = F.interpolate(input_data, scale_factor=bg_scale, mode='bilinear', align_corners=False)
        bg_upblur = opt.bg_base_blur(bg_upblur)

        b, c, h, w = input_data.shape
        ph0, ph1, pw0, pw1 = [0] * 4
        min_sz = 256
        if h < min_sz or w < min_sz:

            ph0 = 0 if h > min_sz else (min_sz - h) // 2
            pw0 = 0 if w > min_sz else (min_sz - w) // 2

            ph1 = ph0
            pw1 = pw0
            if h + ph0 + ph1 < min_sz:
                ph1 += min_sz - h - ph0 - ph1

            if w + pw0 + pw1 < min_sz:
                pw1 += min_sz - w - pw0 - pw1

            input_data = F.pad(input_data, (pw0, pw1, ph0, ph1))

        code, result = svc_get_tensor_result(input_data, bg_scale, 2)

        if result.shape[2] > h * bg_scale or result.shape[3] > w * bg_scale:
            top = ph0 * bg_scale
            bottom = -ph1 * bg_scale
            left = pw0 * bg_scale
            right = -pw1 * bg_scale
            if bottom == 0:
                bottom = ph0 + h * bg_scale
            if right == 0:
                right = pw0 + w * bg_scale

            result = result[:, :, top:bottom, left:right].contiguous()

        result = result + bg_upblur

        img_out = T.ToPILImage()(result.squeeze(0).clamp(0, 1).cpu())
        bn = splitext(basename(fn))[0]
        out_fn = join(output_dir, f"{bn}_2x.png")
        img_out.save(out_fn)
