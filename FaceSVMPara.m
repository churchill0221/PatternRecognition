%% 邱吉尔 1140329098 模式识别Project 人脸识别 SVM参数选择
% 1 读取图片
% 2 PCA降维
% 3 训练SVM识别
clear;clc;
%% 1 read images
% 从1-40文件夹读每个文件夹里面的1-10个文件；
% 对每10个文件进行重排，选择4个做训练数据，6个做测试数据
% for train_num =3:1:5;%训练个数
train_num =9;
train_samples = [];
train_labels = [];
test_samples = [];
test_labels = [];


pathname = 'faceImage\orlData\';

for i = 1:40
    % 随机选取训练和测试数据
    a = 1:10;
    b = randperm(10);
    c = a(b(1:train_num));%训练数据
    d = a(b(train_num+1:10));%测试数据
    
    % 读取测试数据
    for j = a(b(1:train_num))
        imgname = strcat(num2str(i),'\',num2str(j),'.pgm');
        filename = [pathname imgname];
        img = imread(filename);
        [m,n] = size(img);
        img_hist = reshape(img,1,m*n);
        train_samples = [train_samples;img_hist];
        train_labels = [train_labels;i];
    end
    
    % 读取训练数据
    for j = a(b(train_num+1:10))
        imgname = strcat(num2str(i),'\',num2str(j),'.pgm');
        filename = [pathname imgname];
        img = imread(filename);
        [m,n] = size(img);
        img_hist = reshape(img,1,m*n);
        test_samples = [test_samples;img_hist];
        test_labels = [test_labels;i];
    end
end

train_samples = double(train_samples);
test_samples = double(test_samples);

%% PCA降维
[coeff,score,latent,TSQUARED] = princomp(train_samples);

mat_num = 200;
tranMatrix = coeff(:,1:mat_num); %选取50维特征
%训练数据和测试数据降维
S = train_samples * tranMatrix ;
T = test_samples * tranMatrix ;
SVM_ACCURACY = zeros(1,61);

% [bestacc,bestc,bestg] = SVMcg(train_labels,S,-5,10,-5,10,10)
for c_i = -30:30
    c = 2^(c_i);
    %     for ii =1:10
    %% SVM训练和预测
    
    model = svmtrain(train_labels,S, sprintf('-t 0 -c %g -v 10',c));
    c
    %[predict_label, rate, dec_values] = svmpredict(test_labels,T,model);
    SVM_ACCURACY(c_i+31) = SVM_ACCURACY(c_i+31) + model;
    %     end
    %     SVM_ACCURACY(c_i) =SVM_ACCURACY(c_i) / 10;
end
data = -30:30;
plot(data,SVM_ACCURACY,'linewidth',2);
xlabel('X(C=2^X)');
ylabel('准确率');
