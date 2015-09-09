close all;
clear all;
clc;

C_Segments=2;

img_original = imread('lena.png');%读入图像
figure,imshow(img_original),title('原始图像');    %显示原图像
img_gray=rgb2gray(img_original);
figure,imshow(img_gray),title('原始灰度图像');

% 获取图像的长宽
[m,n]=size(img_gray);

% 灰度阈值计算
% T=graythresh(img_gray);
% img_bw=im2bw(img_gray,T);
% figure,imshow(img_bw),title('原始二值图像');

% 将图像进行RGB——3通道分解
A = reshape(img_original(:, :, 1), m*n, 1);    % 将RGB分量各转为kmeans使用的数据格式n行，一样一样本
B = reshape(img_original(:, :, 2), m*n, 1);
C = reshape(img_original(:, :, 3), m*n, 1);
dat = [A B C];  % r g b分量组成样本的特征，每个样本有三个属性值，共width*height个样本
[cRGB C] = kmeans(double(dat), C_Segments,...
    'Distance','city',...
    'emptyaction','singleton',...
    'start','sample');    % 使用聚类算法分为2类
rRGB = reshape(cRGB, m, n);     % 反向转化为图片形式
figure, imshow(label2rgb(rRGB)),title('RGB通道分割结果');   % 显示分割结果

% % 将图像进行单一通道灰度分解
% GraySeg= reshape(img_gray(:, :), m*n, 1);
% cGray=kmeans(double(GraySeg), 2);
% rGray= reshape(cGray, m, n);     % 反向转化为图片形式
% figure, imshow(label2rgb(rGray)),title('灰度通道分割结果');   % 显示分割结果