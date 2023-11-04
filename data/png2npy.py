import os
import argparse
import skimage.io as sio
import numpy as np

parser = argparse.ArgumentParser(description='Pre-processing .png images')
parser.add_argument('--pathFrom', default='F:\datasets\SISR\Train\Train_LR4',
                    help='directory of images to convert')
parser.add_argument('--pathTo', default='F:\datasets\SISR\Train\Train_LR4_',
                    help='directory of images to save')
parser.add_argument('--split', default=True,
                    help='save individual images')
parser.add_argument('--select', default='',
                    help='select certain path')

args = parser.parse_args()

for (path, dirs, files) in os.walk(args.pathFrom):
    # 目录地址  目录下文件夹名  目录下文件名
    print(path)
    targetDir = os.path.join(args.pathTo, path[len(args.pathFrom) + 1:])  # 加个斜杠
    if len(args.select) > 0 and path.find(args.select) == -1:
        continue

    if not os.path.exists(targetDir):
        os.mkdir(targetDir)

    if len(dirs) == 0:
        pack = {}
        n = 0
        for fileName in files:
            (idx, ext) = os.path.splitext(fileName)
            if ext == '.png':
                image = sio.imread(os.path.join(path, fileName))
                if args.split:
                    np.save(os.path.join(targetDir, idx + '.npy'), image)  # 仅仅变更后缀
                n += 1
                if n % 100 == 0:
                    print('Converted ' + str(n) + ' images.')
