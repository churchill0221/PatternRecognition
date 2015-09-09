%% �񼪶� 1140329098 ģʽʶ��Project ����ʶ��
% 1 ��ȡͼƬ
% 2 PCA��ά
% 3 ѵ��SVM,KNN��ϡ��ʶ��
clear;clc;
%% 1 read images
% ��1-40�ļ��ж�ÿ���ļ��������1-10���ļ���
% ��ÿ10���ļ��������ţ�ѡ��4����ѵ�����ݣ�6������������
% for train_num =3:1:5;%ѵ������
train_num =4;
train_samples = [];
train_labels = [];
test_samples = [];
test_labels = [];


pathname = 'faceImage\orlData\';

for i = 1:40
    % ���ѡȡѵ���Ͳ�������
    a = 1:10;
    b = randperm(10);
    c = a(b(1:train_num));%ѵ������
    d = a(b(train_num+1:10));%��������
    
    % ��ȡ��������
    for j = a(b(1:train_num))
        imgname = strcat(num2str(i),'\',num2str(j),'.pgm');
        filename = [pathname imgname];
        img = imread(filename);
        [m,n] = size(img);
        img_hist = reshape(img,1,m*n);
        train_samples = [train_samples;img_hist];
        train_labels = [train_labels;i];
    end
    
    % ��ȡѵ������
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

%% PCA��ά
[coeff,score,latent,TSQUARED] = princomp(train_samples);
% �Լ�д��PCA
% %1.����Э������ 
% COV_ingredients = cov(train_samples);
% %2.��������ֵD����������V.����V�������ж�Ӧ��coeff��D�Խ���������Ӧ��latent
% [V,D] = eig(COV_ingredients);

%��ά����
for mat_num = 50:50:300
% mat_num = 50;
    tranMatrix = coeff(:,1:mat_num); %ѡȡ50ά����
    %ѵ�����ݺͲ������ݽ�ά
    S = train_samples * tranMatrix ;
    T = test_samples * tranMatrix ;
    
    SVM_ACCURACY = 0;
    KNN_ACCURACY = 0;
    SRC_ACCURACY = 0;
    for ii =1:10
        %% SVMѵ����Ԥ��
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
        %% ϡ���ʾ
        
        S_SRC = S';T_SRC = T';%ѵ�����ݺͲ�������
        S_SRC_L = train_labels';T_SRC_L = test_labels';
        pre_label =[];
        for i=1:(10-train_num)*40
            x_out = SolveHomotopy_CBM_std(S_SRC, T_SRC(:,i),'lambda', 0.01);%ϡ�����
            for j=1:40
                mu=zeros(train_num*40,1);
                id=(j==S_SRC_L);%ȡ��ѵ���������ֵ�
                mu(id)=x_out(id);%ȡ��������ϡ��ֵ
                r(j)=norm(T_SRC(:,i)-S_SRC*mu);%�������ƶ�
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
% title('������Ŀ����ȷ�ʹ�ϵ');
% plot(time,svm_accuracy_t,'or',time,knn_accuracy_t,'+g',time,src_accuracy_t,'*b','linewidth',2);
% legend('SVM','KNN','SRC');
% axis([3 5 0 1]);
mat=[50:50:300];
figure;
title('PCAά������ȷ�ʹ�ϵ');
plot(mat,svm_accuracy_t,'or',mat,knn_accuracy_t,'+g',mat,src_accuracy_t,'*b','linewidth',2);
legend('SVM','KNN','SRC');
axis([40 310 0 1]);