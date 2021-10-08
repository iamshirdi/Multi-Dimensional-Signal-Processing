%% circular radial version
clc;close all;clear all
[f1,f2]=freqspace(64)
%freqspace returns the implied frequency range for 
%equally spaced frequency responses. 
[x,y]=meshgrid(f1,f2); 
Hd=zeros(size(x)); 
 r=sqrt(x.^2 + y.^2); 
  d=find(r<0.4);
  Hd(d)=ones(size(d));
  %complete vector directly giving ones of req length
   mesh(f1,f2,Hd) 
   xlabel('f1') 
 ylabel('f2')
figure
freqz2(Hd)
 %%
 function freqz2d(h,dB)
colormap('default')
db=10^(-dB/20);
h=rot90(fliplr(flipud(h)),-1); % rotates the filter mask to matrix coordinates 
H=fft2(h,64,64); % FFT 
ax=-pi:2*pi/64:pi-2*pi/64; 
mesh(ax,ax,20*log10(abs(fftshift(H'))+db)); % FRF of the filter
xlabel('omega1') % X label
ylabel('omega2') % Y label
zlabel('magnitude [dB]') % Z label
 end

 %% seperable Method after windowing impulserespons
%seprable 2d window
clc;close all;clear all
[f1,f2] = freqspace(64);  
[x,y] = meshgrid(f1,f2);
Hd = zeros(size(x));
r = sqrt(x.^2+y.^2);
d = find(r<0.4); 
Hd(d) = ones(size(d));
figure; %rectangle
mesh(f1,f2,Hd) ;
   xlabel('f1') ;
 ylabel('f2');
h1 = fwind1(Hd,hamming(11),hamming(11)); 
h2 = fwind1(Hd,boxcar(11),boxcar(11)); 
figure; 
freqz2(h1); 
title('FRF hamming ');
xlabel('f1'); 
ylabel('f2'); 
figure;
freqz2(h2)
title('FRF rect');
xlabel('f1'); % X label
ylabel('f2'); % Y label
%% CIRCULAR METHOD fourier transform after windowing
% windowing truncating seperable two futions
clc;close all;clear all
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
%% TRANSFORMATION METHOD
clc;close all;clear all
[f1,f2] = freqspace(64);  
[x,y] = meshgrid(f1,f2);
Hd = zeros(size(x));
r = sqrt(x.^2+y.^2);
d = find(r<0.4); 
Hd(d) = ones(size(d)); %impulse 1d
N=11;
h1 = fir1(N-1,0.4,hamming(11));%hanning 1d
h2 = fir1(N-1,0.4,boxcar(11)); %window 1d
4figure
freqz2(h);%window
title('FRF Hamming Transformation')
xlabel('f1'); 
ylabel('f2'); 
%using hamming fft plot
h3 = ftrans2(h2); %transfunction window
figure
freqz2(h3); 
title('FRF Rect Transformation')
xlabel('f1');
ylabel('f2'); 

%% 2b window

% windowing truncating seperable two futions
clc;close all;clear all
[f1,f2] = freqspace(64);  
[x,y] = meshgrid(f1,f2);
Hd = zeros(size(x));
r = sqrt(x.^2+y.^2);
d = find(r>0.2); 
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

%% 2b transform
clc;close all;clear all
[f1,f2] = freqspace(64);  
[x,y] = meshgrid(f1,f2);
Hd = zeros(size(x));
r = sqrt(x.^2+y.^2);
d = find(r>0.2); 
Hd(d) = ones(size(d)); %impulse 1d
N=11;
h1 = fir1(N-1,0.2,'high',hamming(11));%hanning 1d
h2 = fir1(N-1,0.2,'high',boxcar(11)); %window 1d
h = ftrans2(h1); %transfunction default ham
figure
freqz2(h);%window
title('FRF Hamming Transformation')
xlabel('f1'); 
ylabel('f2'); 
%using hamming fft plot

h3 = ftrans2(h2); %transfunction window
figure
freqz2(h3); 
title('FRF Rect Transformation')
xlabel('f1');
ylabel('f2'); 

%% 2d

clc;close all;clear all
% h1 = fir1(N-1,0.5,'low',hamming(N));%hanning 1d
% freqz2(h);%window
% title('FRF Hamming Transformation')
% xlabel('f1'); 
% ylabel('f2'); 
% %using hamming fft plot
%window 1d
% t=[0 0.25 0; -0.25 0 -0.25; 0 0.25 0]
% h = ftrans2(h1,t); %transfunction default ham

N=21;
h2 = fir1(N-1,0.8,boxcar(N)); 
figure
t=[0 0.25 0; -0.25 0 -0.25; 0 0.25 0]

h3 = ftrans2(h2,t); %transfunction window
figure
freqz2(h3); 
title('FRF Rect Transformation')
xlabel('f1');
ylabel('f2'); 





