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
inputnum = i;
tranMatrix = coeff(:,1:inputnum);
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
input_train = S(n(1:lengthfiles*5),:)';
output_train = label(n(1:lengthfiles*5),:);
output_train_NN = output(n(1:lengthfiles*5),:)';
input_test = S(n(lengthfiles*5+1:lengthfiles*10),:)';
output_test = label(n(lengthfiles*5+1:lengthfiles*10),:);
output_test_NN = output(n(lengthfiles*5+1:lengthfiles*10),:)';

%% SVM分类器
model = svmtrain(output_train,input_train');
[predict_label, rate, dec_values] = svmpredict(output_test,input_test',model);

%输入数据归一化
%[inputn,inputps]=mapminmax(input_train);
inputn = input_train;
%% 神经网络结构初始化
innum = inputnum;
midnum = 40;
outnum = 10;

%权值初始化
w1=rands(midnum,innum);
b1=rands(midnum,1);
w2=rands(outnum,midnum);
b2=rands(outnum,1);

w2_1=w2;w2_2=w2_1;
w1_1=w1;w1_2=w1_1;
b1_1=b1;b1_2=b1_1;
b2_1=b2;b2_2=b2_1;

%学习率
xite = 0.1;
alfa = 0.01;

for ii=1:500
    E(ii)=0;
    for i=1:1:250
       %% 正向预测输出 
        x =input_train(:,i);
        y =output_train_NN(:,i);
        
        %隐含层输出
        z2 = w1*x+b1;
        a2 = 1./(1+exp(-z2));
        % 输出层输出
        yn = w2 * a2 + b2;
       %% 反向传播误差
        % 输出层误差
        e3 = y-yn;
        dw2 = e3*a2';
        db2 = e3;
        dfa = a2.*(1-a2);
        e2 = w2'*e3.*dfa;
        dw1 = e2*x';
        db1 = e2;
        
        w1=w1_1+xite*dw1;
        b1=b1_1+xite*db1;
        w2=w2_1+xite*dw2;
        b2=b2_1+xite*db2;
        
        w1_2=w1_1;w1_1=w1;
        w2_2=w2_1;w2_1=w2;
        b1_2=b1_1;b1_1=b1;
        b2_2=b2_1;b2_1=b2;
    end
end

%% 分类
%inputn_test=mapminmax('apply',input_test,inputps);
inputn_test = input_test;
for ii=1:1
    for i=1:250
        %隐含层输出
        for j=1:1:midnum
            I(j)=w1(j,:)*inputn_test(:,i)+b1(j);
            Iout(j)=1/(1+exp(-I(j)));
        end
        fore(:,i)=w2*Iout'+b2;
    end
end

%% 结果分析

for i=1:250
    output_fore(i)=find(fore(:,i)==max(fore(:,i)));
end

%BP网络预测误差
error=output_fore-label(n(251:500))';
accuracy = 0;
for i = 1:250
    if(error(i)==0)
        accuracy = accuracy+1;
    end
end

accuracy = accuracy/250;
fprintf('SVM准确率 %f\n', rate(1)/100);
fprintf('NN准确率 %f\n ', accuracy);