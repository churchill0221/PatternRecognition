clear;clc;
for train_n =3:5
    train_s = [];train_l = [];
    test_s = [];test_l = [];
    pathname = 'faceImage\orlData\';
    for i = 1:40
        a = 1:10;
        c = a(1:train_n);
        d = a(train_n+1:10);
        % 读取测试数据
        for j = a(1:train_n)
            img_name = strcat(num2str(i),'\',num2str(j),'.pgm');
            file_name = [pathname img_name];
            img = imread(file_name);
            [m,n] = size(img);
            img_h = reshape(img,1,m*n);
            train_s = [train_s;img_h];
            train_l = [train_l;i];
        end
        
        % 读取训练数据
        for j = a(train_n+1:10)
            img_name = strcat(num2str(i),'\',num2str(j),'.pgm');
            file_name = [pathname img_name];
            img = imread(file_name);
            [m,n] = size(img);
            img_h = reshape(img,1,m*n);
            test_s = [test_s;img_h];
            test_l = [test_l;i];
        end
    end
    train_s = double(train_s);
    test_s = double(test_s);
    %% PCA
    [coeff,score,latent,TSQUARED] = princomp(train_s);
    
    %降维矩阵
    mat_n = 100
    tranMatrix = coeff(:,1:mat_n); %特征
    %降维
    train_ss = train_s * tranMatrix ;
    test_ss = test_s * tranMatrix ;
    SVM_ACCURACY = 0;
    SRC_ACCURACY = 0;
    
    %% SVM训练和预测
    %     for c_i = -30:30 %C参数选择
    %         c = 2^(c_i);
    %         model = svmtrain(train_labels,S, sprintf('-t 0 -c %g -v 10',c));
    %     end
    model = svmtrain(train_l,train_ss, sprintf('-t 0 -c %g',2^(-5)));
    [predict_label, rate, dec_values] = svmpredict(test_l,test_ss,model);
    SVM_ACCURACY =  rate(3);
    %% 稀疏表示
    S_SRC = train_ss';T_SRC = test_ss';%训练数据和测试数据
    S_SRC_L = train_l';T_SRC_L = test_l';
    pre_label =[];
    for i=1:(10-train_n)*40
        x_out = SolveHomotopy_CBM_std(S_SRC, T_SRC(:,i),'lambda', 0.01);
        for j=1:40
            mu=zeros(train_n*40,1);
            id=(j==S_SRC_L);%取该训练样本的字典
            mu(id)=x_out(id);%取该样本的稀疏值
            R(j)=norm(T_SRC(:,i)-S_SRC*mu);%计算相似度
        end
        [temp,index]=min(R);
        pre_label(i)=index;
    end
    accuracy=sum(pre_label==T_SRC_L)/((10-train_n)*40);
    SRC_ACCURACY =  accuracy;
    
    svm_accuracy_t(train_n-2) = SVM_ACCURACY;
    src_accuracy_t(train_n-2) = SRC_ACCURACY;
end
% time = [3,4,5];
% figure(1)
% plot(time,svm_accuracy_t,'r',time,src_accuracy_t,'b');
m=[3:5];
plot(m,svm_accuracy_t,'r',m,src_accuracy_t,'b');