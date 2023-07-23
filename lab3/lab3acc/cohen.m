function [CD,f,t] = cohen(x,fs,type)
%Function to compute several of Cohen's class of time-frequency
% distributions 
% John T. Semmlow, Biosignal and biomedical image processing: MATLAB-based applications, Part 1
%
%Outputs
%   CD Selected distribution
%   f Frequency vector for plotting
%   t Time vector for plotting
%Inputs
%   x  Complex signal
%   fs Sampling frequency
%   type of distribution. Valid arguments are:
%   'choi' (Choi-Williams), 'BJC' (Born-Jorden-Cohen);
%   and 'R_M' (Rihaczek-Margenau) Default is Wigner-Ville
%
%Assing constants and check input
sigma = 1;                    % Choi-Williams constant
L = 30;                       % Size of determining function
%
[N,xcol] = size(x);
if N < xcol                   % Make signal a column vector if 
 x = x';                      % necessary
 N = xcol;
end
t = (1:N)/fs;                 % Calculate time and frequency
f = (1:N) * (fs/(2*N));       % vectors for plotting
%
% Compute instantenous autocorrelation: Eq. (7)

CD = int_autocorr(x);
if type(1) == 'c'             % Get appropriate determining
                              % function
  G = choi(sigma,L);          % Choi-Williams
elseif type(1) == 'B'
  G = BJC(L);                 % Born-Jorden-Cohen
elseif type(1) == 'R'
  G = R_M(L);                 % Rihaczek-Margenau
else
  G = zeros(L,L);             % Default Wigner-Ville
  G(L/2,L/2) = 1;
end
%
%size(G)
% figure
%  mesh(1:L,1:L,G);        % Plot determining function
%  xlabel('N');ylabel('N');    % and label axis 
%  zlabel('G(N,N)');
%
% Convolve determining function with instantenous 
%  autocorrelation
CD = conv2(CD,G);             % 2-D convolution
CD = CD(1:N,1:N);             % Truncate extra points
                              % produced by convolution
%
% Take FFT again, FFT taken with respect to columns
CD = flipud(fft(CD));         % Output distribution
% CD = fft(CD);         % Output distribution poprawka TB - częstość rośnie w górę
end