clc;
clear all;
close all;
load mandrill; % Load Mandrill Image
I=ind2gray(X,map); % Convert indexed image to intensity image
I=I(1:128,128:128+256-1); % scale the mandrill image to 128*256 pixels image
N1=128; % Signal Dimension
N2=256; % Signal Dimension
w1=2*pi/8; % Frequency
w2=2*pi/4; % Frequency
n1=(0:N1-1)'*ones(1,N2);
n2=ones(N1,1)*(0:N2-1);
x=cos(w1*n1+w2*n2); % cosine signal
In=I+x; % Adding Cosine disturbance to the scaled Image
imshow(mat2gray(In)); % Display the resultant added image
load hlp1 % Load the Lowpass filter-1 .mat file from the MATLAB Directory
Hf=20*log(abs(fftshift(fft2(h,64,64)))+0.01)./2.303; % Frequency resoponse of Lowpass filter-2
ax=-pi:(2*pi)/63:pi;
figure
mesh(ax,ax,Hf);  % Mesh plot of the Filtered response of Undefined signal
Z=filter2(h,In); % Filtering the scaled image with Lowpass filter-1
figure
imshow(mat2gray(Z)); % Display the Filtered Image
load hlp2 % Load the Lowpass filter-2 .mat file from the MATLAB Directory
Hf=20*log(abs(fftshift(fft2(h,64,64)))+0.01)./2.303; % Frequency resoponse of Lowpass filter-2
ax=-pi:(2*pi)/63:pi;
figure
mesh(ax,ax,Hf);  % Mesh plot of the Filtered response of Undefined signal
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
