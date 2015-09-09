%% �񼪶� 1140329098 ģʽʶ��Project ����ʶ�� SVM����ѡ��
% 1 ��ȡͼƬ
% 2 PCA��ά
% 3 ѵ��SVMʶ��
clear;clc;
%% 1 read images
% ��1-40�ļ��ж�ÿ���ļ��������1-10���ļ���
% ��ÿ10���ļ��������ţ�ѡ��4����ѵ�����ݣ�6������������
% for train_num =3:1:5;%ѵ������
train_num =9;
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

mat_num = 200;
tranMatrix = coeff(:,1:mat_num); %ѡȡ50ά����
%ѵ�����ݺͲ������ݽ�ά
S = train_samples * tranMatrix ;
T = test_samples * tranMatrix ;
SVM_ACCURACY = zeros(1,61);

% [bestacc,bestc,bestg] = SVMcg(train_labels,S,-5,10,-5,10,10)
for c_i = -30:30
    c = 2^(c_i);
    %     for ii =1:10
    %% SVMѵ����Ԥ��
    
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
ylabel('׼ȷ��');
