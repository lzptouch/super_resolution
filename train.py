import argparse, os
import time

from thop import profile

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from multiprocessing import freeze_support
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import net361_ as network, utils
# from model import compare as network
from data import DIV2K
from model.loss import FocalFrequencyLoss
import random
from collections import OrderedDict


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=8,
                    help="training batch size")
parser.add_argument("--testBatchSize", type=int, default=16,
                    help="testing batch size")
parser.add_argument("-nEpochs", type=int, default=1000,
                    help="number of epochs to train")
parser.add_argument("--lr", type=float, default=2e-4,
                    help="Learning Rate. Default=2e-4")
parser.add_argument("--step_size", type=int, default=10,
                    help="learning rate decay per N epochs")
parser.add_argument("--gamma", type=int, default=0.9,
                    help="learning rate decay factor for step decay")
parser.add_argument("--cuda", action="store_true", default=False,
                    help="use cuda")
parser.add_argument("--resume", default="", type=str,
                    help="path to checkpoint")
parser.add_argument("--start-epoch", default=1, type=int,
                    help="manual epoch number")
parser.add_argument("--threads", type=int, default=1,
                    help="number of threads for data loading")
# parser.add_argument("--train_hr", type=str, default="../dataset/Train/Train_HR_",   # linux中不要直接传“\"
#                     help='dataset directory')
# parser.add_argument("--train_lr", type=str, default="../dataset/Train/Train_LR{}_",
#                     help='dataset directory')

parser.add_argument("--train_hr", type=str, default="F:\datasets\SISR\Train\Train_HR_",   # linux中不要直接传“\"
                    help='dataset directory')
parser.add_argument("--train_lr", type=str, default="F:\datasets\SISR\Train\Train_LR{}_",
                    help='dataset directory')

parser.add_argument("--n_train", type=int, default=800,
                    help="number of training set")
parser.add_argument("--n_val", type=int, default=1,
                    help="number of validation set")
parser.add_argument("--test_every", type=int, default=1000)
parser.add_argument("--scale", type=int, default=4,
                    help="super-resolution scale")
parser.add_argument("--patch_size", type=int, default=192,
                    help="output patch size")
parser.add_argument("--rgb_range", type=int, default=1,
                    help="maxium value of RGB")
parser.add_argument("--n_colors", type=int, default=3,
                    help="number of color channels to use")
parser.add_argument("--pretrained", default="./checkpoint_x4/best.pth", type=str,
                    help="path to pretrained models")
parser.add_argument("--seed", type=int, default=2333)
parser.add_argument("--isY", action="store_true", default=True)
parser.add_argument("--ext", type=str, default='.npy')
parser.add_argument("--phase", type=str, default='train')

args = parser.parse_args()
if torch.cuda.is_available():
    args.cuda = True
print(args)
torch.backends.cudnn.benchmark = True
# random seed
seed = args.seed
if seed is None:
    seed = random.randint(1, 10000)
print("Ramdom Seed: ", seed)
random.seed(seed)
torch.manual_seed(seed)

cuda = args.cuda
# cuda = False

device = torch.device('cuda' if cuda else 'cpu')

print("===> Loading datasets")

trainset = DIV2K.div2k(args, args.train_hr, args.train_lr)
# testset = Set5_val.DatasetFromFolderVal("../dataset/Valid/Valid_HR",
#                                        "../dataset/Valid/Valid_LR{}".format(args.scale),
#                                        args.scale)

# testset = Set5_val.DatasetFromFolderVal("F:\dataset\SISR\Valid\Valid_HR",
#                                        "F:\dataset\SISR\Valid\Valid_LR{}".format(args.scale),
#                                        args.scale)
training_data_loader = DataLoader(dataset=trainset, num_workers=args.threads, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)
# testing_data_loader = DataLoader(dataset=testset, num_workers=args.threads, batch_size=args.testBatchSize,
#                                  shuffle=False)

print("===> Building models")
args.is_train = True

model = network.HLFA(64,scale=args.scale, alpha=0.5)
# model = network.BSRN()
l1_criterion = nn.L1Loss()
frequency_loss = FocalFrequencyLoss()

print("===> Setting GPU")
if cuda:
    model = model.to(device)
    l1_criterion = l1_criterion.to(device)
    frequency_loss = frequency_loss.to(device)

else:
    model = model.to("cpu")
    l1_criterion = l1_criterion.to("cpu")
    frequency_loss = frequency_loss.to("cpu")

if args.pretrained:
    if os.path.isfile(args.pretrained):
        print("===> loading models '{}'".format(args.pretrained))
        # checkpoint = torch.load(args.pretrained)
        checkpoint = torch.load(args.pretrained, map_location=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        new_state_dcit = OrderedDict()
        for k, v in checkpoint.items():
            if 'module' in k:
                name = k[7:]
            else:
                name = k
            new_state_dcit[name] = v
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in new_state_dcit.items() if k in model_dict}
        for k, v in model_dict.items():
            if k not in pretrained_dict:
                print(k)
        model.load_state_dict(pretrained_dict, strict=True)
    else:
        print("===> no models found at '{}'".format(args.pretrained))
print("===> Setting Optimizer")
optimizer = optim.Adam(model.parameters(), lr=args.lr)


def train(epoch):
    model.train()
    utils.adjust_learning_rate(optimizer, epoch, args.step_size, args.lr, args.gamma)
    print('epoch =', epoch, 'lr = ', optimizer.param_groups[0]['lr'])
    for iteration, (lr_tensor, hr_tensor) in enumerate(training_data_loader, 1):

        if cuda:
            lr_tensor = lr_tensor.to(device)  # ranges from [0, 1]
            hr_tensor = hr_tensor.to(device)  # ranges from [0, 1]
        else:
            lr_tensor = lr_tensor.to("cpu")  # ranges from [0, 1]
            hr_tensor = hr_tensor.to("cpu")  # ranges from [0, 1]
        optimizer.zero_grad()
        sr_tensor = model(lr_tensor)
        loss_l1 = l1_criterion(sr_tensor, hr_tensor)
        # loss_panr = psnr_criterion(sr_tensor, hr_tensor)
        loss_sr = loss_l1

        loss_sr.backward()
        optimizer.step()
        if iteration % 100 == 0:
            print("===> Epoch[{}]({}/{}): Loss: {:.5f}".format(epoch, iteration, len(training_data_loader),
                                                                  loss_sr.item()))


# def valid():
#     model.eval()
#
#     avg_psnr, avg_ssim = 0, 0
#     for batch in testing_data_loader:
#         lr_tensor, hr_tensor = batch[0], batch[1]
#         if args.cuda:
#             lr_tensor = lr_tensor.to(device)
#             hr_tensor = hr_tensor.to(device)
#
#         with torch.no_grad():
#             pre = model(lr_tensor)
#
#         sr_img = utils.tensor2np(pre.detach()[0])
#         gt_img = utils.tensor2np(hr_tensor.detach()[0])
#         crop_size = args.scale
#         cropped_sr_img = utils.shave(sr_img, crop_size)
#         cropped_gt_img = utils.shave(gt_img, crop_size)
#         if args.isY is True:
#             im_label = utils.quantize(sc.rgb2ycbcr(cropped_gt_img)[:, :, 0])
#             im_pre = utils.quantize(sc.rgb2ycbcr(cropped_sr_img)[:, :, 0])
#         else:
#             im_label = cropped_gt_img
#             im_pre = cropped_sr_img
#         avg_psnr += utils.compute_psnr(im_pre, im_label)
#         avg_ssim += utils.compute_ssim(im_pre, im_label)
#     print("===> Valid. psnr: {:.4f}, ssim: {:.4f}".format(avg_psnr / len(testing_data_loader), avg_ssim / len(testing_data_loader)))


def save_checkpoint(epoch):
    model_folder = "checkpoint_x{}/".format(args.scale)
    model_out_path = model_folder + "epoch_{}.pth".format(epoch)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    torch.save(model.state_dict(), model_out_path)
    print("===> Checkpoint saved to {}".format(model_out_path))

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('Total number of parameters: %d' % num_params)

    if cuda:
        input = torch.randn(1, 3, 320, 180).cuda()
    else:
        input = torch.randn(1, 3, 320, 180)

    macs, params = profile(net, inputs=(input,))
    print('mac:{},params:{}'.format(macs,params))


print_network(model)
if __name__ == '__main__':
    freeze_support()
    for epoch in range(args.start_epoch, args.nEpochs + 1):
        # valid()
        start_time = time.time()
        train(epoch)
        time_ = time.time() - start_time
        print("time",time_)
        save_checkpoint(epoch)
