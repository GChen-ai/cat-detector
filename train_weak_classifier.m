%function SVM_model = train_weak_classifier(feature_type, win_size, filename)

% change this path if you install the VOC code elsewhere
addpath([cd '/VOCcode']);

% initialize VOC options
VOCinit;

% load image set
[ids,gt]=textread(sprintf(VOCopts.imgsetpath,"cat_train"),'%s %d');
data1=[];
data2=[];
data3=[];
data4=[];
data5=[];
label=[];
count=0;
for i=1:length(ids)
    im_path=['VOC2007/JPEGImages/' ids{i} '.jpg'];
    img=imread(im_path);
    img=rgb2gray(img);
    img=histeq(img);
    img=double(img)/255;
    rec=PASreadrecord(sprintf(VOCopts.annopath,ids{i}));
    for j=1:length(rec.objects)
        bb=rec.objects(j).bbox;
        cls=rec.objects(j).class;
        im_patch=img(bb(2):bb(4),bb(1):bb(3));
        im_patch=imresize(im_patch,[128,128]);
        
        if strcmp(cls,'cat')
            [f1,f2,f3,f4,f5]=feature_extract(im_patch, 24);
            
            %size(f1)
            %size(f2)
            %size(f3)
            %size(f4)
            %size(f5)
            data1=[data1;f1];
            data2=[data2;f2];
            data3=[data3;f3];
            data4=[data4;f4];
            data5=[data5;f5];
            label=[label; 1 ];
        else
            r=rand;
            if r>0.95
                count=count+1;
                [f1,f2,f3,f4,f5]=feature_extract(im_patch, 24);
                data1=[data1;f1];
                data2=[data2;f2];
                data3=[data3;f3];
                data4=[data4;f4];
                data5=[data5;f5];
                label=[label; -1];
            end
        end
    end
end

SVM_model1 =fitcsvm(data1,label,'Standardize',true,'KernelFunction','RBF','KernelScale','auto');
SVM_model2 =fitcsvm(data2,label,'Standardize',true,'KernelFunction','RBF','KernelScale','auto');
SVM_model3 =fitcsvm(data3,label,'Standardize',true,'KernelFunction','RBF','KernelScale','auto');
SVM_model4 =fitcsvm(data4,label,'Standardize',true,'KernelFunction','RBF','KernelScale','auto');
SVM_model5 =fitcsvm(data5,label,'Standardize',true,'KernelFunction','RBF','KernelScale','auto');
% Save the trained model
save('SVM1','SVM_model1');
save('SVM2','SVM_model2');
save('SVM3','SVM_model3');
save('SVM4','SVM_model4');
save('SVM5','SVM_model5');
%end

