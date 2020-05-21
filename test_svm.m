function [count1,count2,count3,count4,count5,count,ada_count]= test_svm()
addpath([cd '/VOCcode']);

% initialize VOC options
VOCinit;
svm1=load('SVM1.mat');
svm2=load('SVM2.mat');
svm3=load('SVM3.mat');
svm4=load('SVM4.mat');
svm5=load('SVM5.mat');
count1=0;
count2=0;
count3=0;
count4=0;
count5=0;
count=0;
ada_count=0;
alpha_svm=[1.0837 1.1944 1.7663 1.1581 1.2129];
% load test image set fo
[ids_test,gt_test]=textread(sprintf(VOCopts.imgsetpath,"csit5410_test"),'%s %d');
for i=1:length(ids_test)
    im_path=['VOC2007/JPEGImages/' ids_test{i} '.jpg'];
    img=imread(im_path);
    img=rgb2gray(img);
    img=histeq(img);
    img=double(img)/255;
    rec=PASreadrecord(sprintf(VOCopts.annopath,ids_test{i}));
    for j=1:length(rec.objects)
        bb=rec.objects(j).bbox;
        cls=rec.objects(j).class;
        count=count+1;
        if strcmp(cls,'cat')
            label=1;
        else
            label=-1;
        end
        center_x=floor((bb(3)+bb(1))/2);
        center_y=floor((bb(4)+bb(2))/2);
        %bb=[center_x-64 center_y-64 center_x+64 center_y+64];
        
        im_patch=img(bb(2):bb(4),bb(1):bb(3));
        im_patch=imresize(im_patch,[128,128]);
        
        [f1,f2,f3,f4,f5]=feature_extract(im_patch, 24);
        [l1,~]=predict(svm1.SVM_model1,f1);
        if l1==label
            count1=count1+1;
        end
        [l2,~]=predict(svm2.SVM_model2,f2);
        if l2==label
            count2=count2+1;
        end
        [l3,~]=predict(svm3.SVM_model3,f3);
        if l3==label
            count3=count3+1;
        end
        [l4,~]=predict(svm4.SVM_model4,f4);
        if l4==label
            count4=count4+1;
        end
        [l5,~]=predict(svm5.SVM_model5,f5);
        if l5==label
            count5=count5+1;
        end
        ada_result=l1*alpha_svm(1)+l2*alpha_svm(2)+l3*alpha_svm(3)+l4*alpha_svm(4)+l5*alpha_svm(5);
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

end