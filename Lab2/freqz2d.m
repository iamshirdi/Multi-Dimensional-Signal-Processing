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
