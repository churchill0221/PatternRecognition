clear;clc;
%% 字符识别主要包括以下几个部分

%% 导入读取图片

allsamples=[]; tcoor=[];
tic;

for j = 0:9
    pathname = 'C:\学习\课程学习\计算智能\charSamples\';
    filename = strcat(num2str(j));
    pathname = [pathname filename];
    pathname = [pathname '\'];
    files = dir(fullfile(pathname,'*.png'));
    lengthfiles = length(files);
    %lengthfiles = 20;
    for i = 1:lengthfiles
        file_img=[pathname files(i).name]; 
        sample = imread(file_img);
        %[m n] = size(hist_sample);
        m = 20;n=10;
        sample = imresize(sample,[20,10]);
        sample=edge(sample,'zerocross'); %用zerocross算子进行边缘检测
        %imshow(sample)
        bb = sample(1:m*n);
        bb = double(bb);
        allsamples = [allsamples;bb];
    end
end
[coeff,score,latent,tsquared] = pca(allsamples); 

variance_prop = cumsum(latent)./sum(latent);

for i = 1:length(latent)
    if(variance_prop(i) >0.95)
        break;
    end
end
inputn = i;
tranMatrix = coeff(:,1:inputn);
%训练数据和测试数据降维
S = allsamples *tranMatrix ;

aa = ones(lengthfiles,1);bb = ones(lengthfiles,1)*2;cc = ones(lengthfiles,1)*3;
dd = ones(lengthfiles,1)*4;ee = ones(lengthfiles,1)*5;ff = ones(lengthfiles,1)*6;
gg = ones(lengthfiles,1)*7;hh = ones(lengthfiles,1)*8;ii = ones(lengthfiles,1)*9;
kk = ones(lengthfiles,1)*10;
label = [aa;bb;cc;dd;ee;ff;gg;hh;ii;kk];

for i=1:lengthfiles*10
    switch label(i)
        case 1
            output(i,:)=[1 0 0 0 0 0 0 0 0 0];
        case 2
            output(i,:)=[0 1 0 0 0 0 0 0 0 0];
        case 3
            output(i,:)=[0 0 1 0 0 0 0 0 0 0];
        case 4
            output(i,:)=[0 0 0 1 0 0 0 0 0 0];
        case 5
            output(i,:)=[0 0 0 0 1 0 0 0 0 0];
        case 6
            output(i,:)=[0 0 0 0 0 1 0 0 0 0];
        case 7
            output(i,:)=[0 0 0 0 0 0 1 0 0 0];
        case 8
            output(i,:)=[0 0 0 0 0 0 0 1 0 0];
        case 9
            output(i,:)=[0 0 0 1 0 0 0 0 1 0];
        case 10
            output(i,:)=[0 0 0 1 0 0 0 0 0 1];
    end
end

k = rand(1,lengthfiles*10);
[m,n] = sort(k);
input_train = S(n(1:lengthfiles*5),:);
%output_train = label(n(1:lengthfiles*5),:);
output_train = label(n(1:lengthfiles*5),:);
input_test = S(n(lengthfiles*5+1:lengthfiles*10),:);
output_test = label(n(lengthfiles*5+1:lengthfiles*10),:);
%SVM分类器
model = svmtrain(output_train,input_train);
[predict_label, rate, dec_values] = svmpredict(output_test,input_test,model);

% traindata = S;
% trainlabel = label;
% 
% model = svmtrain(trainlabel,traindata);
% [predict_label, rate, dec_values] = svmpredict(trainlabel,S,model);



%matlab自带神经网络分类器

%选连样本输入输出数据归一化
[inputn,inputps]=mapminmax(input_train');
[outputn,outputps]=mapminmax(output_train',1,10);

net = newff(inputn,outputn,i+1,{ 'logsig' 'purelin' } , 'traingdx'); %构建神经网络分类器

net.trainparam.show = 50 ;
net.trainparam.epochs = 5000 ;
net.trainparam.goal = 0.01 ;
net.trainParam.lr = 0.01 ;

net = train(net,inputn,outputn);
%% BP网络预测
%预测数据归一化
inputn_test=mapminmax('apply',input_test',inputps);
 
%网络预测输出
an=sim(net,inputn_test);
 
%网络输出反归一化
BPoutput=mapminmax(an,1,10);

BPoutput1 = ceil(BPoutput);

