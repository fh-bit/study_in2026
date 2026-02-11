% 3D Fourier Spectral Laplacian Solver (Using ndgrid)
clear; close all; clc;
L = 2*pi;     
N = 32;       
dx = L/N;
xyz = dx * (0:N-1);  
[X, Y, Z] = ndgrid(xyz, xyz, xyz);
u = sin(X) .* cos(Y) .* sin(Z);
lap_exact = -3 * u;
k_vec = [0:N/2-1, -N/2:-1]'; 
[Kx, Ky, Kz] = ndgrid(k_vec, k_vec, k_vec);
u_hat = fftn(u);
L_k = -(Kx.^2 + Ky.^2 + Kz.^2);
lap_hat = L_k .* u_hat;
lap_numerical = real(ifftn(lap_hat));
error = abs(lap_numerical - lap_exact);
max_error = max(error(:));

fprintf('Grid Size: %d x %d x %d\n', N, N, N);
fprintf('Max Error: %.4e\n', max_error);