
function freqz2d(h,dB)
colormap('default')
db=10^(-dB/20);
h=rot90(fliplr(flipud(h)),-1); % rotates the filter mask to matrix coordinates 
H=fft2(h,64,64); % FFT 
ax=-pi:2*pi/64:pi-2*pi/64; 
mesh(ax,ax,20*log10(abs(fftshift(H'))+db)); % FRF of the filter
xlabel('omega1------------->') % X label
ylabel('omega2------------->') % Y label
zlabel('magnitude [dB]------------>') % Z label
end

------------------------------------------Task 1----------------------------------------

clc;
clear all;
close all;
[f1,f2] = freqspace(64);  % Frequency Spacing
[x,y] = meshgrid(f1,f2);
Hd = zeros(size(x));
r = sqrt(x.^2+y.^2);
d = find(r<0.4); % Given condition
Hd(d) = ones(size(d));
figure;
mesh(f1,f2,Hd); % Plotting the frequency reponse function
title('Frequency Response Function');
xlabel('f1--------------->'); % X label
ylabel('f2--------------->'); % Y label

%% Linear Separable Method


h1 = fwind1(Hd,hamming(11),hamming(11)); % FRF of FIR filter using linear separable method (Hamming window)
h2 = fwind1(Hd,boxcar(11),boxcar(11)); % FRF of FIR filter using linear separable method (Rectangular or Boxcar window)
figure;
subplot(1,2,1);
freqz2d(h1,40); % Plot the 2-D Frequency Response Function which renders a floor of -40 dB
title('FRF of FIR filter using linear separable method (Hamming window)');
xlabel('omega1--------------->'); % X label
ylabel('omega2--------------->'); % Y label
figure;
subplot(1,2,2);
freqz2d(h2,40); % Plot the 2-D Frequency Response Function which renders a floor of -40 dB
title('FRF of FIR filter using linear separable method (Rectangular or Boxcar window)');
xlabel('omega1--------------->'); % X label
ylabel('omega2--------------->'); % Y label
%% CIRCULAR METHOD

h3 = fwind1(Hd,hamming(11)); % FRF of FIR filter using circular method (Hamming window)
h4 = fwind1(Hd,boxcar(11)); % FRF of FIR filter using circular method (Rectangular or Boxcar window)
figure;
freqz2d(h3,40); % Plot the 2-D Frequency Response Function which renders a floor of -40 dB
title('FRF of FIR filter using circular method (Hamming window)');
xlabel('omega1--------------->'); % X label
ylabel('omega2--------------->'); % Y label
figure;
freqz2d(h4,40); % Plot the 2-D Frequency Response Function which renders a floor of -40 dB
title('FRF of FIR filter using circular method (Rectangular or Boxcar window)');
xlabel('omega1--------------->'); % X label
ylabel('omega2--------------->'); % Y label
%% TRANSFORMATION METHOD

N=11;
h5 = fir1(N-1,0.4,boxcar(11));
h6 = fir1(N-1,0.4,boxcar(11));
h5_t = ftrans2(h5);
figure
freqz2d(h5_t,40); % Plot the 2-D Frequency Response Function which renders a floor of -40 dB
title('FRF of FIR filter using Transformation method (Hamming window)')
xlabel('omega1--------------->'); % X label
ylabel('omega2--------------->'); % Y label
h6_t = ftrans2(h5);
figure
freqz2d(h6_t,40); % Plot the 2-D Frequency Response Function which renders a floor of -40 dB
title('FRF of FIR filter using Transformation method (Rectangular or Boxcar window)')
xlabel('omega1--------------->'); % X label
ylabel('omega2--------------->'); % Y label







------------------------------------------Task 2a--------------------------------

clc;
clear all;
close all;
[f1,f2] = freqspace(64);  % Frequency Spacing
[x,y] = meshgrid(f1,f2);
Hd = zeros(size(x));
r = sqrt(x.^2+y.^2);
d = find(r<0.4); % Given condition
Hd(d) = ones(size(d));
figure;
mesh(f1,f2,Hd); % Plotting the frequency reponse function
title('Frequency Response Function');
xlabel('f1--------------->'); % X label
ylabel('f2--------------->'); % Y label

%% Linear Separable Method


h1 = fwind1(Hd,hamming(11),hamming(11)); % FRF of FIR filter using linear separable method (Hamming window)
h2 = fwind1(Hd,boxcar(11),boxcar(11)); % FRF of FIR filter using linear separable method (Rectangular or Boxcar window)
figure;
subplot(1,2,1);
freqz2d(h1,40); % Plot the 2-D Frequency Response Function which renders a floor of -40 dB
title('FRF of FIR filter using linear separable method (Hamming window)');
xlabel('omega1--------------->'); % X label
ylabel('omega2--------------->'); % Y label
figure;
subplot(1,2,2);
freqz2d(h2,40); % Plot the 2-D Frequency Response Function which renders a floor of -40 dB
title('FRF of FIR filter using linear separable method (Rectangular or Boxcar window)');
xlabel('omega1--------------->'); % X label
ylabel('omega2--------------->'); % Y label
%% CIRCULAR METHOD

h3 = fwind1(Hd,hamming(11)); % FRF of FIR filter using circular method (Hamming window)
h4 = fwind1(Hd,boxcar(11)); % FRF of FIR filter using circular method (Rectangular or Boxcar window)
figure;
subplot(1,2,1);
freqz2d(h3,40); % Plot the 2-D Frequency Response Function which renders a floor of -40 dB
title('FRF of FIR filter using circular method (Hamming window)');
xlabel('omega1--------------->'); % X label
ylabel('omega2--------------->'); % Y label
figure;
subplot(1,2,2);
freqz2d(h4,40); % Plot the 2-D Frequency Response Function which renders a floor of -40 dB
title('FRF of FIR filter using circular method (Rectangular or Boxcar window)');
xlabel('omega1--------------->'); % X label
ylabel('omega2--------------->'); % Y label
%% TRANSFORMATION METHOD

N=11;
h5 = fir1(N-1,0.4,boxcar(11));
h6 = fir1(N-1,0.4,boxcar(11));
h5_t = ftrans2(h5);
figure
subplot(1,2,1);
freqz2d(h5_t,40); % Plot the 2-D Frequency Response Function which renders a floor of -40 dB
title('FRF of FIR filter using Transformation method (Hamming window)')
xlabel('omega1--------------->'); % X label
ylabel('omega2--------------->'); % Y label
h6_t = ftrans2(h5);
figure
subplot(1,2,2);
freqz2d(h6_t,40); % Plot the 2-D Frequency Response Function which renders a floor of -40 dB
title('FRF of FIR filter using Transformation method (Rectangular or Boxcar window)')
xlabel('omega1--------------->'); % X label
ylabel('omega2--------------->'); % Y label






---------------------------------------Task 2b--------------------------------

clc;
clear all;
close all;
[f1,f2] = freqspace(64);  % Frequency Spacing
[x,y] = meshgrid(f1,f2);
Hd = zeros(size(x));
r = sqrt(x.^2+y.^2);
d = find(r>0.2); % Given condition
Hd(d) = ones(size(d));
figure;
mesh(f1,f2,Hd); % Plotting the frequency reponse function
title('Frequency Response Function');
xlabel('f1--------------->'); % X label
ylabel('f2--------------->'); % Y label

%%  Linear Separable Method


h1 = fwind1(Hd,hamming(11),hamming(11)); % FRF of FIR filter using linear separable method (Hamming window)
h2 = fwind1(Hd,boxcar(11),boxcar(11)); % FRF of FIR filter using linear separable method (Rectangular or Boxcar window)
figure;
subplot(1,2,1);
freqz2d(h1,40); % Plot the 2-D Frequency Response Function which renders a floor of -40 dB
title('FRF of FIR filter using linear separable method (Hamming window)');
xlabel('omega1--------------->'); % X label
ylabel('omega2--------------->'); % Y label
figure;
subplot(1,2,2);
freqz2d(h2,40); % Plot the 2-D Frequency Response Function which renders a floor of -40 dB
title('FRF of FIR filter using linear separable method (Rectangular or Boxcar window)');
xlabel('omega1--------------->'); % X label
ylabel('omega2--------------->'); % Y label
%% CIRCULAR METHOD

h3 = fwind1(Hd,hamming(11)); % FRF of FIR filter using circular method (Hamming window)
h4 = fwind1(Hd,boxcar(11)); % FRF of FIR filter using circular method (Rectangular or Boxcar window)
figure;
subplot(1,2,1);
freqz2d(h3,40); % Plot the 2-D Frequency Response Function which renders a floor of -40 dB
title('FRF of FIR filter using circular method (Hamming window)');
xlabel('omega1--------------->'); % X label
ylabel('omega2--------------->'); % Y label
figure;
subplot(1,2,2);
freqz2d(h4,40); % Plot the 2-D Frequency Response Function which renders a floor of -40 dB
title('FRF of FIR filter using circular method (Rectangular or Boxcar window)');
xlabel('omega1--------------->'); % X label
ylabel('omega2--------------->'); % Y label
%% TRANSFORMATION METHOD

N=11;
h5 = fir1(N-1,0.2,'high',boxcar(11));
h6 = fir1(N-1,0.2,'high',boxcar(11));
h5_t = ftrans2(h5);
figure
subplot(1,2,1);
freqz2d(h5_t,40); % Plot the 2-D Frequency Response Function which renders a floor of -40 dB
title('FRF of FIR filter using Transformation method (Hamming window)')
xlabel('omega1--------------->'); % X label
ylabel('omega2--------------->'); % Y label
h6_t = ftrans2(h5);
figure
subplot(1,2,2);
freqz2d(h6_t,40); % Plot the 2-D Frequency Response Function which renders a floor of -40 dB
title('FRF of FIR filter using Transformation method (Rectangular or Boxcar window)')
xlabel('omega1--------------->'); % X label
ylabel('omega2--------------->'); % Y label





-----------------------------------------------Task_2c------------------------------------

clc;
clear all;
close all;
[f1,f2] = freqspace(64);  % Frequency Spacing
[x,y] = meshgrid(f1,f2);
Hd = zeros(size(x));
r = sqrt(x.^2+y.^2);
d = find(r>0.3 & r<0.6); % Given condition
Hd(d) = ones(size(d));
figure;
mesh(f1,f2,Hd); % Plotting the frequency reponse function
title('Frequency Response Function');
xlabel('f1--------------->'); % X label
ylabel('f2--------------->'); % Y label

%%  Linear Separable Method


h1 = fwind1(Hd,hamming(11),hamming(11)); % FRF of FIR filter using linear separable method (Hamming window)
h2 = fwind1(Hd,boxcar(11),boxcar(11)); % FRF of FIR filter using linear separable method (Rectangular or Boxcar window)
figure;
subplot(1,2,1);
freqz2d(h1,40); % Plot the 2-D Frequency Response Function which renders a floor of -40 dB
title('FRF of FIR filter using linear separable method (Hamming window)');
xlabel('omega1--------------->'); % X label
ylabel('omega2--------------->'); % Y label
figure;
subplot(1,2,2);
freqz2d(h2,40); % Plot the 2-D Frequency Response Function which renders a floor of -40 dB
title('FRF of FIR filter using linear separable method (Rectangular or Boxcar window)');
xlabel('omega1--------------->'); % X label
ylabel('omega2--------------->'); % Y label
%% CIRCULAR METHOD

h3 = fwind1(Hd,hamming(11)); % FRF of FIR filter using circular method (Hamming window)
h4 = fwind1(Hd,boxcar(11)); % FRF of FIR filter using circular method (Rectangular or Boxcar window)
figure;
subplot(1,2,1);
freqz2d(h3,40); % Plot the 2-D Frequency Response Function which renders a floor of -40 dB
title('FRF of FIR filter using circular method (Hamming window)');
xlabel('omega1--------------->'); % X label
ylabel('omega2--------------->'); % Y label
figure;
subplot(1,2,2);
freqz2d(h4,40); % Plot the 2-D Frequency Response Function which renders a floor of -40 dB
title('FRF of FIR filter using circular method (Rectangular or Boxcar window)');
xlabel('omega1--------------->'); % X label
ylabel('omega2--------------->'); % Y label
%% TRANSFORMATION METHOD

N=11;
h5 = fir1(N-1,[0.3 0.6],boxcar(11));
h6 = fir1(N-1,[0.3 0.6],boxcar(11));
h5_t = ftrans2(h5);
figure
subplot(1,2,1);
freqz2d(h5_t,40); % Plot the 2-D Frequency Response Function which renders a floor of -40 dB
title('FRF of FIR filter using Transformation method (Hamming window)')
xlabel('omega1--------------->'); % X label
ylabel('omega2--------------->'); % Y label
h6_t = ftrans2(h5);
figure
subplot(1,2,2);
freqz2d(h6_t,40); % Plot the 2-D Frequency Response Function which renders a floor of -40 dB
title('FRF of FIR filter using Transformation method (Rectangular or Boxcar window)')
xlabel('omega1--------------->'); % X label
ylabel('omega2--------------->'); % Y label





-----------------------------------------Task2d-----------------------------------------

%% 

clc;
clear all;
close all;
sequence = [0 0.25 0; -0.25 0 -0.25; 0 0.25 0];
N=21; % Dimension
h1 = fir1(N-1,0.5,hamming(N)); % FRF of FIR filter using Transformation method  
H1 = fft(h1,512); % FFT of impulse response of FIR Filter
figure
plot(0:1/256:1,20*log10(abs(H1(1:257)))); % Plot the 1-D Frequency Response Function
h = ftrans2(h1,sequence);
figure
freqz2d(h,40);% Plot the 2-D Frequency Response Function which renders a floor of -40 dB
title('Output of 2-D 21*21 FIF Filter');
xlabel('omega1--------------->'); % X label
ylabel('omega2--------------->'); % Y label