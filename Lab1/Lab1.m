%% pre assignment 1 Convolution

%data 2d in  space origin to matrix 1 conversion
clc;close all;clear all
 n1x=0:3;n2x=0:3;
[xx,yx]=meshgrid(n1x,n2x);
x=ones(length(n1x),length(n2x));
stem3(xx,yx,x); %input signal

n1h=-2:1;n2h=0:1;
[xh,yh]=meshgrid(n1h,n2h);
h=zeros(size(xh));
 h(1,1:4)=[5 6 7 8]; 
 h(2,1:4)=[1 2 3 4]; %colunms in matrix rows in space
  figure
stem3(xh,yh,h) %impulse signal
 y=conv2(x,h) %zero paddding conv
 
 display(y)
  n2o=0-2:3-2+2-1;n1o=0-2:3-2+4-1;
[xo,yo]=meshgrid(n1o,n2o);
figure
stem3(xo,yo,y);
% %seperability square shape with input sepration 4*1 and 1*4
%can we seperate input signal instead of impulse for compuatations
% y1=conv2(x(1,1:4),h)
% y2=conv2(x(1:4,1),y1)

%% Task 1 zero padding changes everything 
%impulse orgigin center
clc;clear all;close all;
x = ones(4); % Input signal
h = [5 6 7 8;1 2 3 4]; % Impulse response
c=conv2(x,h,'full')
cinput=conv2(x,h,'same') %performs by keeping origin at center without rotating
%The filter2 function filters data by taking the 2-D convolution 
%of the input X and the coefficient matrix H rotated 180 degrees.
%Specifically, filter2(H,X,shape) is equivalent to conv2(X,rot90(H,2),shape).
%B = rot90(A,k) rotates array A counterclockwise by k*90 degrees, where k is an integer.
%180 degrees clock and anticlock both same
f=filter2(h,x)




%% pre assignment 2 Fourier Transform

%% pra 2a
clc;close all;clear all
N1=16;
N2=8;
x=ones(N1,N2);
F1=zeros(N1);
F2=zeros(N2);
for k1=0:N1-1
    for n1=0:N1-1
         F1(k1+1,n1+1)=exp(-j*2*pi*k1*n1/N1); %twiddle factor
    end
end

for k2=0:N2-1            
    for n2=0:N2-1
        F2(k2+1,n2+1)=exp(-j*2*pi*k2*n2/N2);
        end;
end;

X=F1*x*F2;
Xa=abs(X)
X2=fft2(x)

%% pra 2b
N1=8;
N2=8;
w=ones(N1,N2);
W=fft2(w)

%% pra 2c
clc;close all;clear all
N1=8;
N2=8;
l1=1;l2=2;
w1=2*pi*l1/(N1);
w2=2*pi*l2/(N2);
n1=0:N1-1;n2=0:N2-1;
[n1,n2]=meshgrid(n1,n2)
x=cos(w1*n1+w2*n2)









%% task 2 fourier transform

%% 2a
clc;close all;clear all
N1=16;
N2=8;
x=ones(N1,N2);
F1=zeros(N1);
F2=zeros(N2);
for k1=0:N1-1
    for n1=0:N1-1
         F1(k1+1,n1+1)=exp(-j*2*pi*k1*n1/N1); %twiddle factor
    end
end

for k2=0:N2-1            
    for n2=0:N2-1
        F2(k2+1,n2+1)=exp(-j*2*pi*k2*n2/N2);
        end;
end;

X=F1*x*F2;
Xa=abs(X)
X2=fft2(x)

%% 2b
clc;close all;clear all
N1=8;
N2=8;
x=ones(N1,N2);
X=fft2(x,64,64);
ax=-pi:2*pi/63:pi;
colormap('default')
mesh(ax,ax,20*log10(abs(fftshift(X'))+0.01)) 
% the constant 0.01 will render a floor at -40dB
xlabel('omega1')
ylabel('omega2')


%% filter response
clc;close all;clear all
load hlp1.mat
N1=9;N2=9;
H=fft2(h,64,64)
ax=-pi:2*pi/63:pi;
colormap('default')
mesh(ax,ax,20*log10(abs(fftshift(H'))+0.01)) 
% the constant 0.01 will render a floor at -40dB
xlabel('omega1')
ylabel('omega2')


%% 2c

clc;close all;clear all
N1=64;
N2=64;
l1=8;l2=16;
w1=2*pi*l1/(N1);
w2=2*pi*l2/(N2);
n1=0:N1-1;n2=0:N2-1;
[n1,n2]=meshgrid(n1,n2)
x=cos(w1*n1+w2*n2)

figure
imshow(x); 
A=fft2(x); 
figure
mesh(20*log10(abs(fftshift(A'))+0.01));

%%
clc;close all;clear all
N1=8;
N2=8;
l1=1;l2=2;
w1=2*pi*l1/(N1);
w2=2*pi*l2/(N2);
n1=0:N1-1;n2=0:N2-1;
[n1,n2]=meshgrid(n1,n2)
x=cos(w1*n1+w2*n2)

figure
imshow(x); 
A=fft2(x,64,64); 
figure
mesh(20*log10(abs(fftshift(A'))+0.01));

%%
clc;close all;clear all
N1=16;
N2=16;
l1=2;l2=4;
w1=2*pi*l1/(N1);
w2=2*pi*l2/(N2);
n1=0:N1-1;n2=0:N2-1;
[n1,n2]=meshgrid(n1,n2)
x=cos(w1*n1+w2*n2)

figure
imshow(x); 
A=fft2(x,64,64); 
figure
mesh(20*log10(abs(fftshift(A'))+0.01));

%% 2d
clc;clear all;close all;
load mandrill;
I=ind2gray(X,map);
I=I(1:128,120:120+256-1) %scaled by 128 to 256
%to get middle eyes started at 120
figure
imshow(I);%original image scaled
Ib=fft2(I).*exp(j*2*pi*rand(128,256));
inv=real(ifft2(Ib))
% inv=mat2gray(inv);
figure
imshow(mat2gray(inv))
%amplitude magnitude change
If=fft2(I)./abs(fft2(I));
invf=real(ifft2(If))
% inv=mat2gray(inv);
figure
imshow(mat2gray(invf))


%phase
%% 3 filtering

%% 3a
clc;close all;clear all
N1=480;
N2=500;
l1=1;l2=2;
w1=2*pi*l1/(N1);
w2=2*pi*l2/(N2);
n1=0:N1-1;n2=0:N2-1;
[n1,n2]=meshgrid(n1,n2);
xn=cos(w1*n1+w2*n2); %signal cosine

% load hlp1.mat
load hlp2.mat
N1=8;N2=8;
Hh=fft2(h,64,64)
ax=-pi:2*pi/63:pi;
colormap('default')
mesh(ax,ax,20*log10(abs(fftshift(Hh'))+0.01)) %filter response
% the constant 0.01 will render a floor at -40dB
xlabel('omega1')
ylabel('omega2')

load mandrill
 I=ind2gray(X,map); %grayscale
 a=I;
o=filter2(h,a);
figure
imshow(mat2gray(o)) %intesntity matrix

%% 
clc;
clear all;
close all;
load mandrill
figure 
I=ind2gray(X,map); 
I=I(1:128,120:120+256-1); 
imshow(I); 
load hlp1.mat 
Z=filter2(h,I);
figure
imshow(mat2gray(Z)); 
load hlp2.mat 
Z1=filter2(h,I); 
figure
imshow(mat2gray(Z1));
load hhp1.mat 
Z2=filter2(h,I);
figure
imshow(mat2gray(Z2)); 
load hhp2.mat
Z3=filter2(h,I); 
figure
imshow(mat2gray(Z3));
load hk.mat 
Z4=filter2(h,I); 
figure
imshow(mat2gray(Z4));

%%

clc;
clear all;
close all;
load mandrill; 
I=ind2gray(X,map);
I=I(1:128,128:128+256-1); 
N1=128; 
N2=256; 
w1=2*pi/8; 
w2=2*pi/4; 
n1=(0:N1-1)'*ones(1,N2);
n2=ones(N1,1)*(0:N2-1);
x=cos(w1*n1+w2*n2);
In=I+x; 
imshow(mat2gray(In)); 
load hlp1
Hf=20*log(abs(fftshift(fft2(h,64,64)))+0.01)./2.303; 
ax=-pi:(2*pi)/63:pi;
figure
mesh(ax,ax,Hf);  
Z=filter2(h,In); 
figure
imshow(mat2gray(Z)); 
load hlp2 
Hf=20*log(abs(fftshift(fft2(h,64,64)))+0.01)./2.303; 
ax=-pi:(2*pi)/63:pi;
figure
mesh(ax,ax,Hf);  
Z=filter2(h,In); 
figure
imshow(mat2gray(Z)); 
load hhp1 
Hf=20*log(abs(fftshift(fft2(h,64,64)))+0.01)./2.303; 
ax=-pi:(2*pi)/63:pi;
figure
mesh(ax,ax,Hf);  
Z=filter2(h,In); 
figure
imshow(mat2gray(Z)); 
load hhp2 
Hf=20*log(abs(fftshift(fft2(h,64,64)))+0.01)./2.303;
ax=-pi:(2*pi)/63:pi;
figure
mesh(ax,ax,Hf); 
Z=filter2(h,In); 
figure
imshow(mat2gray(Z)); 
load hk 
Hf=20*log(abs(fftshift(fft2(h,64,64)))+0.01)./2.303;
ax=-pi:(2*pi)/63:pi;
figure
mesh(ax,ax,Hf); 
Z=filter2(h,In); 
figure
imshow(mat2gray(Z)); 
