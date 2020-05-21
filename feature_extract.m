function [fea1,fea2,fea3,fea4,fea5] = feature_extract(I, winsize)
% Complete task 1 here
    [H,W]=size(I);
    ii=zeros([H,W]);
    s=zeros([H,W]);
    s(1:H,1)=double(I(1:H,1));
    ii(1,1)=s(1,1);
    for j=2:W
        s(1,j)=s(1,j-1)+double(I(1,j));
        ii(1,j)=s(1,j);
    end
    for i=2:H
        for j=2:W
            s(i,j)=s(i,j-1)+double(I(i,j));
            ii(i,j)=ii(i-1,j)+s(i,j);
        end
    end
    
    fea1=[];
    fea2=[];
    fea3=[];
    fea4=[];
    fea5=[];
    for i=1:winsize:H-winsize
        for j=1:winsize:W-winsize
                [fea_1,fea_2,fea_3]=haar_same(ii(i:i+winsize,j:j+winsize),winsize,18,18);
                fea_4=haar_notsame(ii(i:i+winsize,j:j+winsize),winsize,21,7);
                fea_5=haar_notsame(ii(i:i+winsize,j:j+winsize),winsize,7,21);
                fea1=[fea1 fea_1];
                fea2=[fea2 fea_2];
                fea3=[fea3 fea_3];
                fea4=[fea4 fea_4];
                fea5=[fea5 fea_5];
            
        end
    end
end
function [haar_feature1,haar_feature2,haar_feature3] = haar_same(II, winsize,h,w)
    H=winsize;
    W=winsize;
    X=floor(W/w);
    Y=floor(H/h);
    haar_feature1=[];
    haar_feature2=[];
    haar_feature3=[];
    for  i=1:h:X
        for j=1:w:Y
			for x=1:W-i*w+1
                for y=1:H-j*h+1
                    white=II(y+j*h,x+w*i)+II(y,x)-II(y,x+w*i)-II(y+j*h,x);
                    black=II(y+j*h,x+floor(w*i/2))+II(y,x)-II(y,floor(x+w*i/2))-II(y+j*h,x);
                    haar_feature1=[haar_feature1 white-black];

                    white=II(y+j*h,x+w*i)+II(y+floor(j*h/2),x)-II(y+floor(j*h/2),x+w*i)-II(y+j*h,x);
                    black=II(y+floor(j*h/2),x+w*i)+II(y,x)-II(y+floor(j*h/2),x+w*i)-II(y,x+w*i);
                    haar_feature2=[haar_feature2 white-black];

                    white=2*II(y+floor(j*h/2),x+floor(i*w/2))+II(y+j*h,x+i*w)-II(y+floor(j*h/2),x+i*w)-II(y+j*h,x+floor(i*w/2))+II(y,x)-II(y,x+floor(i*w/2))-II(floor(y+j*h/2),x);
                    black=II(y+floor(j*h/2),x+i*w)+II(y,x+floor(i*w/2))-II(y,x+i*w)-2*II(y+floor(j*h/2),x+floor(i*w/2))+II(y+j*h,x+floor(i*w/2))+II(y+floor(j*h/2),x)-II(y+j*h,x);
                    haar_feature3=[haar_feature3 white-black];
                end
            end
        end
    end
end


function haar_feature = haar_notsame(II, winsize,h,w)
    H=winsize;
    W=winsize;
    X=floor(W/w);
    Y=floor(H/h);
    haar_feature=[];
    for  i=1:w:X
        for j=1:h:Y
			for x=1:2:W-i*w+1
                for y=1:2:H-j*h+1
                    if h==21
                        white=II(y+j*h,x+i*w)+II(y+floor(2/3*j*h),x)-II(y+floor(2/3*j*h),x+i*w)-II(y+j*h,x)+II(y+floor(1/3*j*h),x+i*w)+II(x,y)-II(y,x+i*w)-II(y+floor(1/3*j*h),x);
                        black=II(y+floor(2/3*j*h),x+i*w)+II(y+floor(1/3*j*h),x)-II(y+floor(1/3*j*h),x+i*w)-II(y+floor(2/3*j*h),x);
                        haar_feature=[haar_feature white-black];
                    else
                        white=II(y+j*h,x+i*w)+II(y,x+floor(2/3*i*w))-II(y+j*h,x+floor(2/3*i*w))-II(y,x+i*w)+II(y+j*h,x+floor(1/3*i*w))+II(x,y)-II(y+j*h,x)-II(y,x+floor(1/3*i*w));
                        black=II(y+j*h,x+floor(2/3*i*w))+II(y,x+floor(1/3*w*i))-II(y+j*h,x+floor(1/3*i*w))-II(y,x+floor(2/3*i*w));
                        haar_feature=[haar_feature white-black];
                    end
                end
            end
        end
    end


end