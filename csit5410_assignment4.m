close all;
clear;

% change this path if you install the VOC code elsewhere
addpath([cd '/VOCcode']);

% initialize VOC options
VOCinit;

% load image set for Adaboost 
[ids,gt]=textread(sprintf(VOCopts.imgsetpath,"cat_val"),'%s %d');




[ids_test,gt_test]=textread(sprintf(VOCopts.imgsetpath,"csit5410_test"),'%s %d');
tic
%Complete Task 3.1 - 3.3 here
[count1,count2,count3,count4,count5,count,ada]=test_svm();
disp(['Correctness (Weak Classifier 1):' num2str(count1) '/' num2str(count) ' on csit5410_test']);
disp(['Correctness (Weak Classifier 2):' num2str(count2) '/' num2str(count) ' on csit5410_test']);
disp(['Correctness (Weak Classifier 3):' num2str(count3) '/' num2str(count) ' on csit5410_test']);
disp(['Correctness (Weak Classifier 4):' num2str(count4) '/' num2str(count) ' on csit5410_test']);
disp(['Correctness (Weak Classifier 5):' num2str(count5) '/' num2str(count) ' on csit5410_test']);

disp(['Correctness (Strong Classifier):' num2str(ada) '/' num2str(count) ' on csit5410_test']);
%adaboost
svm1=load('SVM1.mat');
svm2=load('SVM2.mat');
svm3=load('SVM3.mat');
svm4=load('SVM4.mat');
svm5=load('SVM5.mat');
val_num=0;
count1=0;
count2=0;
count3=0;
count4=0;
count5=0;
e1=[];
e2=[];
e3=[];
e4=[];
e5=[];
for i=1:length(ids)
    im_path=['VOC2007/JPEGImages/' ids{i} '.jpg'];
    rec=PASreadrecord(sprintf(VOCopts.annopath,ids{i}));
    val_num=val_num+length(rec.objects);
    img=imread(im_path);
    img=rgb2gray(img);
    img=histeq(img);
    img=double(img)/255;
    for j=1:length(rec.objects)
        bb=rec.objects(j).bbox;
        cls=rec.objects(j).class;
        if strcmp(cls,'cat')
            label=1;
        else
            label=-1;
        end
        im_patch=img(bb(2):bb(4),bb(1):bb(3));
        im_patch=imresize(im_patch,[128,128]);
        
        [f1,f2,f3,f4,f5]=feature_extract(im_patch, 24);
        [l1,~]=predict(svm1.SVM_model1,f1);
        if l1==label
            count1=count1+1;
            e1=[e1; 1];
        else
            e1=[e1; -1];
        end
        [l2,~]=predict(svm2.SVM_model2,f2);
        if l2==label
            count2=count2+1;
            e2=[e2; 1];
        else
            e2=[e2; -1];
        end
        [l3,~]=predict(svm3.SVM_model3,f3);
        if l3==label
            count3=count3+1;
            e3=[e3; 1];
        else
            e3=[e3; -1];
        end
        [l4,~]=predict(svm4.SVM_model4,f4);
        if l4==label
            count4=count4+1;
            e4=[e4; 1];
        else
            e4=[e4; -1];
        end
        [l5,~]=predict(svm5.SVM_model5,f5);
        if l5==label
            count5=count5+1;
            e5=[e5; 1];
        else
            e5=[e5; -1];
        end
    end
end
acc=[count1/val_num count2/val_num count3/val_num count4/val_num count5/val_num];
e=[e1 e2 e3 e4 e5];
% val_num=7818;
% 
% acc=load('acc.mat')
% acc=acc.acc
% e=load('err.mat')
% e=e.e
[~,idx]=sort(acc,'descend');
w=ones([val_num,1]);
alpha_svm=zeros([5,1]);
for i=1:5
    w=w/sum(w);
    error_k=sum(w.*(1-e(:,idx(i)))/2);
    alpha_svm(idx(i))=0.5*log((1-error_k)/error_k);
    w=w*exp(-alpha_svm(idx(i))*e(idx(i)));
end
alpha_svm
ada_count=0;
for i=1:length(ids)
    im_path=['VOC2007/JPEGImages/' ids{i} '.jpg'];
    rec=PASreadrecord(sprintf(VOCopts.annopath,ids{i}));
    val_num=val_num+length(rec.objects);
    img=imread(im_path);
    img=rgb2gray(img);
    img=histeq(img);
    img=double(img)/255;
    for j=1:length(rec.objects)
        bb=rec.objects(j).bbox;
        cls=rec.objects(j).class;
        if strcmp(cls,'cat')
            label=1;
        else
            label=-1;
        end
        im_patch=img(bb(2):bb(4),bb(1):bb(3));
        im_patch=imresize(im_patch,[128,128]);
        
        [f1,f2,f3,f4,f5]=feature_extract(im_patch, 24);
        [p1,~]=predict(svm1.SVM_model1,f1);
        [p2,~]=predict(svm2.SVM_model2,f2);
        [p3,~]=predict(svm3.SVM_model3,f3);
        [p4,~]=predict(svm4.SVM_model4,f4);
        [p5,~]=predict(svm5.SVM_model5,f5);
        ada_result=p1*alpha_svm(1)+p2*alpha_svm(2)+p3*alpha_svm(3)+p4*alpha_svm(4)+p5*alpha_svm(5);
        if ada_result>=0.5
            pre=1;
        else
            pre=-1;
        end
        if pre==label
            ada_count=ada_count+1;
        end

    end
end
disp(['Correctness (Strong Classifier):' num2str(ada_count) '/' num2str(val_num) ' on cat_val']);
File = dir(fullfile('test_images/','*.jpg'));
FileNames = {File.name}';
for k = 1:size(FileNames,1)
    img_path=strcat('test_images/', char(FileNames(k)));
    test_img=imread(img_path);
    img_gray=rgb2gray(test_img);
    img_gray=histeq(img_gray);
    [H,W]=size(img_gray);
    img_gray=double(img_gray)/255;

    bbox=[];
    mid_bbox=[];
    for i=1:16:H-128+1
        for j=1:16:W-128+1
            img_patch=img_gray(i:i+127,j:j+127);
            [f1,f2,f3,f4,f5]=feature_extract(img_patch, 24);
            [p1,~]=predict(svm1.SVM_model1,f1);
            [p2,~]=predict(svm2.SVM_model2,f2);
            [p3,~]=predict(svm3.SVM_model3,f3);
            [p4,~]=predict(svm4.SVM_model4,f4);
            [p5,~]=predict(svm5.SVM_model5,f5);
            ada_result=p1*alpha_svm(1)+p2*alpha_svm(2)+p3*alpha_svm(3)+p4*alpha_svm(4)+p5*alpha_svm(5);
            if ada_result>=0.5
                mid_bbox=[mid_bbox;i i+127 j j+127 ada_result];
            end
        end
    end
    nms=NMS_det(mid_bbox);
    bbox=[bbox;nms];
    big_bbox=[];
    for i=1:16:H-256+1
        for j=1:16:W-256+1
            img_patch=img_gray(i:i+255,j:j+255);
            img_patch=imresize(img_patch,[128,128]);
            [f1,f2,f3,f4,f5]=feature_extract(img_patch, 24);
            [p1,~]=predict(svm1.SVM_model1,f1);
            [p2,~]=predict(svm2.SVM_model2,f2);
            [p3,~]=predict(svm3.SVM_model3,f3);
            [p4,~]=predict(svm4.SVM_model4,f4);
            [p5,~]=predict(svm5.SVM_model5,f5);
            ada_result=p1*alpha_svm(1)+p2*alpha_svm(2)+p3*alpha_svm(3)+p4*alpha_svm(4)+p5*alpha_svm(5);
            if ada_result>=0.5
                big_bbox=[big_bbox;i i+255 j j+255 ada_result];
            end
        end
    end
    nms=NMS_det(big_bbox);
    bbox=[bbox;nms];
    small_bbox=[];
    for i=1:8:H-64+1
        for j=1:8:W-64+1
            img_patch=img_gray(i:i+63,j:j+63);
            img_patch=imresize(img_patch,[128,128]);
            [f1,f2,f3,f4,f5]=feature_extract(img_patch, 24);
            [p1,~]=predict(svm1.SVM_model1,f1);
            [p2,~]=predict(svm2.SVM_model2,f2);
            [p3,~]=predict(svm3.SVM_model3,f3);
            [p4,~]=predict(svm4.SVM_model4,f4);
            [p5,~]=predict(svm5.SVM_model5,f5);
            ada_result=p1*alpha_svm(1)+p2*alpha_svm(2)+p3*alpha_svm(3)+p4*alpha_svm(4)+p5*alpha_svm(5);
            if ada_result>=0.5
                small_bbox=[small_bbox;i i+63 j j+63 ada_result];
            end
        end
    end
    nms=NMS_det(small_bbox);
    bbox=[bbox;nms];
    bbox=sortrows(bbox,5);
    for i=1:size(bbox,1)
        test_img(bbox(i,1):bbox(i,2),bbox(i,3),1)=255;
        test_img(bbox(i,1):bbox(i,2),bbox(i,3),2)=255;
        test_img(bbox(i,1):bbox(i,2),bbox(i,3),3)=0;
        
        test_img(bbox(i,1):bbox(i,2),bbox(i,4),1)=255;
        test_img(bbox(i,1):bbox(i,2),bbox(i,4),2)=255;
        test_img(bbox(i,1):bbox(i,2),bbox(i,4),3)=0;


        test_img(bbox(i,1),bbox(i,3):bbox(i,4),1)=255;
        test_img(bbox(i,1),bbox(i,3):bbox(i,4),2)=255;
        test_img(bbox(i,1),bbox(i,3):bbox(i,4),3)=0;


        test_img(bbox(i,2),bbox(i,3):bbox(i,4),1)=255;
        test_img(bbox(i,2),bbox(i,3):bbox(i,4),2)=255;
        test_img(bbox(i,2),bbox(i,3):bbox(i,4),3)=0;
        test_img=insertText(test_img,[bbox(i,3) bbox(i,1)],bbox(i,5));
        if i==3
            break
        end
    end
    figure()
    imshow(test_img)
end
toc