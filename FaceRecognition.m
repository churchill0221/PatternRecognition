%% 邱吉尔 1140329098 模式识别Project 人脸识别
% 1 读取图片
% 2 PCA降维
% 3 训练SVM,KNN和稀疏识别
clear;clc;
%% 1 read images
% 从1-40文件夹读每个文件夹里面的1-10个文件；
% 对每10个文件进行重排，选择4个做训练数据，6个做测试数据
% for train_num =3:1:5;%训练个数
train_num =4;
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
% 自己写的PCA
% %1.计算协方差阵 
% COV_ingredients = cov(train_samples);
% %2.计算特征值D，特征向量V.其中V降序排列对应着coeff，D对角线排序后对应着latent
% [V,D] = eig(COV_ingredients);

%降维矩阵
for mat_num = 50:50:300
% mat_num = 50;
    tranMatrix = coeff(:,1:mat_num); %选取50维特征
    %训练数据和测试数据降维
    S = train_samples * tranMatrix ;
    T = test_samples * tranMatrix ;
    
    SVM_ACCURACY = 0;
    KNN_ACCURACY = 0;
    SRC_ACCURACY = 0;
    for ii =1:10
        %% SVM训练和预测
        model = svmtrain(train_labels,S, sprintf('-t 0 -c %g',2^(-5)));
        [predict_label, rate, dec_values] = svmpredict(test_labels,T,model);
        SVM_ACCURACY = SVM_ACCURACY + rate(3);
        %% KNN
        class = knnclassify(T,S,train_labels,10);
        accuracy_knn = 0;
        % acc_knn = class-test_labels;
        % for i = 1:240
        %     if(acc_knn(i)==0)
        %         accuracy_knn = accuracy_knn + 1;
        %     end
        % end
        accuracy_knn = sum(class==test_labels)/((10-train_num)*40);
        KNN_ACCURACY = KNN_ACCURACY + accuracy_knn;
        %% 稀疏表示
        
        S_SRC = S';T_SRC = T';%训练数据和测试数据
        S_SRC_L = train_labels';T_SRC_L = test_labels';
        pre_label =[];
        for i=1:(10-train_num)*40
            x_out = SolveHomotopy_CBM_std(S_SRC, T_SRC(:,i),'lambda', 0.01);%稀疏矩阵
            for j=1:40
                mu=zeros(train_num*40,1);
                id=(j==S_SRC_L);%取该训练样本的字典
                mu(id)=x_out(id);%取该样本的稀疏值
                r(j)=norm(T_SRC(:,i)-S_SRC*mu);%计算相似度
            end
            [temp,index]=min(r);
            pre_label(i)=index;
        end
        accuracy=sum(pre_label==T_SRC_L)/((10-train_num)*40);
        SRC_ACCURACY = SRC_ACCURACY + accuracy;
    end
    SVM_ACCURACY =SVM_ACCURACY / 10
    KNN_ACCURACY =KNN_ACCURACY / 10
    SRC_ACCURACY =SRC_ACCURACY / 10
    %     svm_accuracy_t(train_num-2) = SVM_ACCURACY;
    %     knn_accuracy_t(train_num-2) = KNN_ACCURACY;
    %     src_accuracy_t(train_num-2) = SRC_ACCURACY;
    
    %%
    svm_accuracy_t(mat_num/50) = SVM_ACCURACY;
    knn_accuracy_t(mat_num/50) = KNN_ACCURACY;
    src_accuracy_t(mat_num/50) = SRC_ACCURACY;
end
% time = [3,4,5];
% figure(1)
% title('测试数目和正确率关系');
% plot(time,svm_accuracy_t,'or',time,knn_accuracy_t,'+g',time,src_accuracy_t,'*b','linewidth',2);
% legend('SVM','KNN','SRC');
% axis([3 5 0 1]);
mat=[50:50:300];
figure;
title('PCA维数和正确率关系');
plot(mat,svm_accuracy_t,'or',mat,knn_accuracy_t,'+g',mat,src_accuracy_t,'*b','linewidth',2);
legend('SVM','KNN','SRC');
axis([40 310 0 1]);