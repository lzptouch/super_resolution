import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import argparse
import torch
import numpy as np
import skimage.color as sc
import cv2
from model import net361_ as network, utils

# Testing settings


parser = argparse.ArgumentParser()
parser.add_argument("--test_hr_folder", type=str, default='F:\datasets\SISR\Test\Set5_HR',
                    help='the folder of the target images')
parser.add_argument("--test_lr_folder", type=str, default='F:\datasets\SISR\Test\Set5_LR4',
                    help='the folder of the input images')

# parser.add_argument("--test_hr_folder", type=str, default=r'E:\dataset\SISR\Test\BSDS100_HR',
#                     help='the folder of the target images')
# parser.add_argument("--test_lr_folder", type=str, default=r'E:\dataset\SISR\Test\BSDS100_LR4',
#                     help='the folder of the input images')

parser.add_argument("--output_folder", type=str, default='./results/Set5')
# parser.add_argument("--checkpoint", type=str, default='./checkpoint_x4/weight36/1/epoch_1.pth',
#                     help='checkpoint folder to use')
parser.add_argument("--checkpoint", type=str, default='./checkpoint_x4/epoch_315.pth',
                    help='checkpoint folder to use')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='use cuda')
parser.add_argument("--upscale_factor", type=int, default=3,
                    help='upscaling factor')
parser.add_argument("--is_y", action='store_true', default=True,
                    help='evaluate on y channel, if False evaluate on RGB channels')
opt = parser.parse_args()



print(opt)
if torch.cuda.is_available():
    opt.cuda = True
else:
    opt.cuda=False
cuda = opt.cuda
device = torch.device('cuda' if cuda else 'cpu')

filepath = opt.test_hr_folder

ext = '.bmp'
# ext = '.png'

filelist = utils.get_list(filepath, ext=ext)
psnr_list = np.zeros(len(filelist))
ssim_list = np.zeros(len(filelist))
time_list = np.zeros(len(filelist))


model = network.HLFA(64,4,0.5)
# model = network.BSRN()
# model_dict = utils.load_state_dict(opt.checkpoint)


psnr = []
for checkpoint in range(1,440):
    opt.checkpoint = r'E:\5.机器学习project\图像超分\模型设计\light_super_resolution\checkpoint_x4/epoch_' +str(checkpoint)+ '.pth'

    if torch.cuda.is_available():
        model_dict = torch.load(opt.checkpoint)
    else:
        model_dict = torch.load(opt.checkpoint, map_location=lambda storage, loc: storage)
    model.load_state_dict(model_dict, strict=True)

    i = 0

    for imname in filelist:
        im_gt = cv2.imread(imname, cv2.IMREAD_COLOR)[:, :, [2, 1, 0]]  # BGR to RGB
        im_gt = utils.modcrop(im_gt, opt.upscale_factor)
        im_l = cv2.imread(opt.test_lr_folder +"\\"+ imname.split('\\')[-1].split('.')[0] + ext, cv2.IMREAD_COLOR)[:, :, [2, 1, 0]]  # BGR to RGB
        if len(im_gt.shape) < 3:
            im_gt = im_gt[..., np.newaxis]
            im_gt = np.concatenate([im_gt] * 3, 2)
            im_l = im_l[..., np.newaxis]
            im_l = np.concatenate([im_l] * 3, 2)
        im_input = im_l / 255.0
        im_input = np.transpose(im_input, (2, 0, 1))
        im_input = im_input[np.newaxis, ...]
        im_input = torch.from_numpy(im_input).float()

        if cuda:
            model = model.to(device)
            im_input = im_input.to(device)

        with torch.no_grad():
            out = model(im_input)

        out_img = utils.tensor2np(out.detach()[0])
        crop_size = opt.upscale_factor
        cropped_sr_img = utils.shave(out_img, crop_size)
        cropped_gt_img = utils.shave(im_gt, crop_size)
        if opt.is_y is True:
            im_label = utils.quantize(sc.rgb2ycbcr(cropped_gt_img)[:, :, 0])
            im_pre = utils.quantize(sc.rgb2ycbcr(cropped_sr_img)[:, :, 0])
        else:
            im_label = cropped_gt_img
            im_pre = cropped_sr_img
        psnr_list[i] = utils.compute_psnr(im_pre, im_label)
        ssim_list[i] = utils.compute_ssim(im_pre, im_label)

        output_folder = os.path.join(opt.output_folder,
                                     imname.split('\\')[-1].split('.')[0] + '.png')

        if not os.path.exists(opt.output_folder):
            os.makedirs(opt.output_folder)

        cv2.imwrite(output_folder, out_img[:, :, [2, 1, 0]])
        i += 1

    print("epoch: {},Mean PSNR: {}, SSIM: {}, TIME: {} ms".format(checkpoint,np.mean(psnr_list), np.mean(ssim_list), np.mean(time_list)))

    psnr.append(np.mean(psnr_list))
print("max PSNR: {}, epoch_number: {}, ".format(np.max(psnr), np.argmax(psnr)))


a=np.array(psnr)
np.save("scripts/net363.npy", a)
