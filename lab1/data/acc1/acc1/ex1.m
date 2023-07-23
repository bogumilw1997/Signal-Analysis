% *******************************************************
% FFT peak search example
% Markus Nentwig 2007
% This program is in the public domain
% Note: Peak search as shown works with basic waveforms
% such as sine, triangle, square, sawtooth etc, but is not a 
% good general pitch detection algorithm 
% *******************************************************

% read wave file
close all; clear all;
[y, fs]=audioread('chord.wav');
[n, nChan]=size(y);
if nChan > 1
  error('this example requires a mono audio file');
end

% calculate the frequency corresponding to each FFT bin.
% this includes negative frequencies!
freqbase=fs*(mod(((0:n-1)+floor(n/2)), n)-floor(n/2))/n;

% apply window and get FFT. "Hamming" could be omitted
spectrum=fft(y .* hamming(n)); 
plot(fft(y));

% get power (square of bins)
spectrum=spectrum .* conj(spectrum);

% pick out the audible range (for plotting)
bin1=min(find(freqbase > 16));
bin2=min(find(freqbase > 16000)-1);
spectrum=spectrum(bin1:bin2);
freqbase=freqbase(bin1:bin2);

% plot. Empty bin: use -150 dB, can't take log(0).
plot(freqbase, 10*log10(spectrum+1e-15));
xlabel('f/Hz'); 
ylabel('p/dB');

% peak search. Note: comparing floats by == is OK here. 
% 'find' can return several results.
peakbin=find(spectrum==max(spectrum));

% look up the frequency corresponding to the bin
fPeak=freqbase(peakbin)

% For comparison: The fundamental pitch of note.wav.
% keyhole C => 9 semitones below 440 Hz "A"
fActual_Hz=440*power(2, -9/12)


