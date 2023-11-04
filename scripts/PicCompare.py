# 将不同的图片切割相同的区域 保存展示并 计算ssim 和psnr值
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
from model import utils
import skimage.color as sc
import cv2
import argparse

def modcrop(im, site, size):
    ims = im[site:site+size, site:site+size, ...]
    return ims

# 设定获取图片的大小和位置坐标 图像名称
parser = argparse.ArgumentParser(description='IMDN')
parser.add_argument("--size", type=int, default=128,)
parser.add_argument("--name",type=str,default="img_030.png")
parser.add_argument("--site",type=int,default=128)
# parser.add_argument("--LR_image_path", type=str,default=r'E:\dataset\SISR\Test\Set5_LR4')
parser.add_argument("--GROUND_target_path", type=str,default=r'F:\datasets\SISR\Test\Urban100_HR')
parser.add_argument("--BICUBIC_target_path", type=str,default=r'F:\datasets\SISR\Test\Urban100_bic4')
parser.add_argument("--save_path", type=str,default="./compare")
opt = parser.parse_args()

dir_path =[r'F:\datasets\SISR\Test\Urban100_bic4',
              r"E:\5.机器学习project\图像超分\经典模型\大模型\复现完成\FSRCNN-pytorch-master\results\test\Urban100\x4",
              r"E:\5.机器学习project\图像超分\经典模型\大模型\复现完成\DRRN-pytorch-master\results\Urban100\x4",
              r"E:\5.机器学习project\图像超分\经典模型\大模型\复现完成\VDSR-PyTorch-master\results\test\Urban100\x4",
              r"E:\5.机器学习project\图像超分\经典模型\大模型\复现完成\IMDN-master\results\IMDN_results\IMDN_x4\Urban100\x4",
              r"E:\5.机器学习project\图像超分\经典模型\轻量化\BSRN-main\results\Urban100\x4",
              r"E:\5.机器学习project\图像超分\模型设计\light_super_resolution\results\x4\Urban100"
              ]




for i in range(len(dir_path)):
    path = dir_path[i]
    target_path = os.path.join(opt.GROUND_target_path,opt.name)
    pre_path = os.path.join(path,opt.name)
    # 获取自己生成的图像 以及其他n张图像的位置 读取图像
    if i == 4 :
        pre_path = os.path.join(path, "img_030x4.png")
    if i == 6:
        pre_path = os.path.join(path, "img_030.png")

    groud_image = np.array(cv2.imread(target_path, cv2.IMREAD_COLOR)[:, :, [2, 1, 0]])
    groud_image[:,:,0]
    img = np.zeros((1020, 672, 3), np.uint8)



    # 起点和终点的坐标
    ptLeftTop = (opt.site, opt.site)
    ptRightBottom = (opt.site+opt.size, opt.site+opt.size)
    point_color = (255, 1, 1)  # BGR
    thickness =2
    lineType = 4
    cv2.rectangle(img, ptLeftTop, ptRightBottom, point_color, thickness, lineType)


    tr_ = img == 255
    fa_ = img == 1
    groud_image[tr_]=255
    groud_image[fa_]=0




    cv2.imwrite(opt.save_path + "/ground" + opt.name, groud_image[:, :, [2, 1, 0]])


    cv_img = cv2.imdecode(np.fromfile(target_path, dtype=np.uint8), -1)
    # im decode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
    target = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)


    cv_img = cv2.imdecode(np.fromfile(pre_path, dtype=np.uint8), -1)
    # im decode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
    pred = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)

    # target = cv2.imread(target_path, cv2.IMREAD_COLOR)[:, :, [2, 1, 0]]  # BGR to RGB
    # pred = cv2.imread(pre_path, cv2.IMREAD_COLOR)[:, :, [2, 1, 0]]

    # 图像裁剪并保存
    croped_target = modcrop(target, opt.site, opt.size)
    croped_out = modcrop(pred, opt.site, opt.size)

    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)


    cv2.imwrite(opt.save_path + "/target" + opt.name.split(".")[0]+".tiff", croped_target[:, :, [2, 1, 0]])
    cv2.imwrite(opt.save_path+"/"+str(i)+ opt.name.split(".")[0]+".tiff", croped_out[:, :, [2, 1, 0]])

    # 提取流明
    target_label = utils.quantize(sc.rgb2ycbcr(croped_target)[:, :, 0])
    out_pre = utils.quantize(sc.rgb2ycbcr(croped_out)[:, :, 0])

    # 计算psnr和ssim的大小
    psnr_out = utils.compute_psnr(out_pre, target_label)
    ssim_out = utils.compute_ssim(out_pre, target_label)
    print(str(i)+"Mean PSNR: {}, SSIM: {}, ".format(psnr_out, ssim_out))
