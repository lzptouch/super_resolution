
from skimage import io
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
from glob import glob
from ntpath import basename
# from scipy.misc import imread
# from skimage.measure import compare_ssim
# from skimage.measure import compare_psnr//
from PIL import Image
from imageio import imread,imwrite

def parse_args():
    parser = argparse.ArgumentParser(description='script to compute all statistics')
    parser.add_argument('--data_from',default=r'F:\datasets\SISR\Train\Flickr2K_HR',type=str)
    parser.add_argument('--data_to',default=r'F:\datasets\SISR\Train\Train_HR' , type=str)
    parser.add_argument('--max_scale' , default=12, type=int)
    parser.add_argument('--debug', default=0, help='Debug', type=int)
    args = parser.parse_args()
    return args
args = parse_args()

for root, dirs ,files,in os.walk(args.data_from):
    for dir in dirs:
        files = list(glob(args.data_from + '/'+dir + '/*.bmp')) + list(glob(args.data_from + '/'+dir +  '/*.png'))
    for fn in sorted(files):
        name = basename(str(fn))

        img_from= imread(args.data_from + '/' + basename(str(fn)))
        w,h = img_from.shape[0:2]

        new_w = w//args.max_scale * args.max_scale
        new_h = h//args.max_scale * args.max_scale
        img_to = img_from[0:new_w,0:new_h,:]

        path = os.path.join(args.data_to, name)
        imwrite(path, img_to)
