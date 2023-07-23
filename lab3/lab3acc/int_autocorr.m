function Rx = int_autocorr(x)
% Function to compute the instantenous autocorrelation
% John T. Semmlow, Biosignal and biomedical image processing: MATLAB-based applications, Part 1
% Output
%   Rx instantaneous autocorrelation
% Input
%   x signal
%
[N, xcol] = size(x);
Rx = zeros(N,N);       % Initialize output
%
% Compute instantenous autocorrelation
for ti = 1:N % Increment over time
  taumax = min([ti-1, N-ti,round(N/2)-1]);
  tau = -taumax:taumax;
  Rx(tau-tau(1)+1,ti) = x(ti+tau).*conj(x(ti-tau));
end
end
