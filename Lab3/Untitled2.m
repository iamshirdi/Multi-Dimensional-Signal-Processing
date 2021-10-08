%%
clc;close all;clear all
a=zeros(20,20);

for i=8:13
    for j=8:13
        a(i,j)=1;
    end
end

imshow(a)
b=medfilt2(a,[3 3]); %default 3 by 3
figure
imshow(b)

%%
clc;close all;clear all
load trees
I=ind2gray(X,map);
J=imnoise(I,'salt & pepper', 0.05);
imshow(J)
b=medfilt2(J,[3 3]); %default 3 by 3
figure
imshow(b)

%%
clc;close all;clear all
% windowing truncating seperable two futions
[f1,f2] = freqspace(64);  
[x,y] = meshgrid(f1,f2);
Hd = zeros(size(x));
r = sqrt(x.^2+y.^2);
d = find(r<0.4); 
Hd(d) = ones(size(d));
h3 = fwind1(Hd,hamming(11)); 
h4 = fwind1(Hd,boxcar(11)); 
figure;
freqz2(h3); title('FRF hamming circular');
xlabel('f1'); 
ylabel('f2'); 
figure;
freqz2(h4); 
title('FRF  rect circular');
xlabel('f1'); 
ylabel('f2'); 
load trees
I=ind2gray(X,map);
J=imnoise(I,'salt & pepper', 0.05);
figure
imshow(J)
b=filter2(h3,J); %default 3 by 3
figure
imshow(b)

%%


