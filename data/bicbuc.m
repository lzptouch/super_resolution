clear;close all;
%% settings
hr_folder = 'F:\datasets\SISR\Test\Set5_LR4';
lr_folder = 'F:\datasets\SISR\Test\Set5_bic4';
scale = 4;

%% generate data
filepaths = dir(fullfile(hr_folder,'*.bmp'));

for i = 1 : length(filepaths)        
    im_gt = imread(fullfile(hr_folder,filepaths(i).name));
    [w,h,c] = size(im_gt)

    %new_w = floor(w/scale)*scale
    %new_h = floor(h/scale)*scale

    %im_lr = imresize(im_gt, [new_w,new_h], 'bicubic');
    im_lr = imresize(im_gt, scale, 'bicubic');
    imwrite(im_lr, fullfile(lr_folder,filepaths(i).name));
end
