clc;
clear all;
close all;
load('Image4.mat');    % loading data of original image
I=IMAGE;
I=mat2gray(I);     
imshow(I)        % original image
title('corrupted image')

%% histogram
imhist(I)

%%  noise remove
h1 = [0 -1 0; -1 5 -1; 0 -1 0];  % High Pass Filter_1
I1 = mat2gray(filter2(h1,I));% Filtering the Image through High pass filter_1
 I1=wiener2(I1);
figure
a=imshow(mat2gray(I1));
title('removed noise')

%  h=fspecial('gaussian',[7,7]);
% I1=imfilter(I,h);
% a=imshow(mat2gray(I1))

%%  Deblurring 
N=11;
[z1,z2]=freqspace(64);  %two-dimensional frequency vectors f1 and f2 for an 64-by-64 matrix.
[c,d]=meshgrid(z1,z2);  % representing on a grid
H=zeros(size(c));      % generating a zeros matrix of the size w1
r=sqrt(c.^2+d.^2);
d=find(r<0.7);        %finding the values which satisfy the condition
H(d)=ones(size(d)); %  generating a ones matrix of the size d

 h1=fwind1(H,hamming(N),hamming(N));
 figure(4)
 

 freqz2(h1)
 B=inverseFilter(I1,h1,1);      
DB2=mat2gray(B);         
figure(5)
imshow(DB2)
title('blurring remove')
 






%% Unwarping 

W=zeros(480,480);
R0=294;
for i=1:380
    for j=1:380
        x_ = j - 190.5;
        y_ = 190.5 - i;
  %taking absolute values of x and y
        x = abs(x_);
        y = abs(y_);
    % determining polar coordinates     
         t = abs(atan(y/x));
        r_ = sqrt(x^2+y^2);
        r = abs(R0.*(asin((r_/R0))));  %given condition
         
        
       %Converting Polar Coordinates (r,theta) to Cartesian Coordinates (x,y)
        
         x = r.*cos(t) ;
         y = r.*sin(t) ;
        if x_ < 0
            x = -x;
        end
        
        if y_ < 0
            y = -y;
        end
        %towards positive infinity 
        i_= (ceil(300- y));  
        j_ = (ceil(x +300));
        W(i_,j_) = DB2(i,j);    
    end
end
figure
imshow(mat2gray(W));  %unwarping image 
title('Unwarped Image')
% W1=medfilt2(W);  
% figure
% imshow(W1);
% title('unwarping first level')
% W2=medfilt2(W1);
% figure; 
% imshow(W2);
% title('unwarping second level')
