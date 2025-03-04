clc
clear variables
close all
% 0 FDM exponential based, 1 FFT method, 2 Spectral differentiation,
% 3 fast sine transform method
technique = 1;
Npa = [51 51]; % number of points per axis (must be odd for FFT method)
Q = 0.1*eye(2); % covariance matrix for dynamics
dt = 0.1; % Numerilam methofs time step (technique 2 and 3)

[timeOut, meanPercentErrorOut, covPercentErrorOut, klOut] = ...
    LGbF_diffusions_2D(technique, Npa, Q, dt); % Call  to calculation

fprintf('Time complexity of time-update %f \n', timeOut)
fprintf('Percentage error of PMD mean approximation %f \n', meanPercentErrorOut)
fprintf('Percentage error of PMD covariance approximation %f \n', covPercentErrorOut)
fprintf('Kullback–Leibler divergence of time-update PMD and true PDF %f \n', klOut)
