% ========================================================
% 2D Fourier Spectral Laplacian Solver
% Code Ref: Standard Tensor Product Implementation
% ========================================================

clear; close all; clc;

%% 1. Grid Generation (2D)
Lx = 2*pi; Ly = 2*pi;
Nx = 64;   Ny = 64;  % Grid points

dx = Lx/Nx; dy = Ly/Ny;
x = dx * (0:Nx-1);   % Row vector
y = dy * (0:Ny-1);   % Row vector
[X, Y] = meshgrid(x, y); % X, Y are 2D matrices

%% 2. Test Function
% u = sin(x)cos(y)
% Exact Laplacian: -2sin(x)cos(y)
u = sin(X) .* cos(Y);
lap_exact = -2 * sin(X) .* cos(Y);

%% 3. Wavenumber Matrices Construction (The Critical Step)
% Construct 1D wavenumbers first
kx = [0:Nx/2-1, -Nx/2:-1]'; % Column vector for consistency logic below
ky = [0:Ny/2-1, -Ny/2:-1]'; 

% Map 1D wavenumbers to 2D Grid
% kx corresponds to the 2nd dimension (columns) -> Meshgrid X
% ky corresponds to the 1st dimension (rows)    -> Meshgrid Y
[Kx, Ky] = meshgrid(kx, ky); 
% Note: Using meshgrid(kx, ky) makes Kx vary across columns, 
% and Ky vary across rows. This matches the [X, Y] orientation.

%% 4. Spectral Calculation
% Step 1: Forward FFT (2D)
u_hat = fft2(u);

% Step 2: The Multiplier for Laplacian -(kx^2 + ky^2)
% Laplacian Symbol L_k
L_k = -(Kx.^2 + Ky.^2);

% Step 3: Compute Derivative in Spectral Space
lap_hat = L_k .* u_hat;

% Step 4: Inverse FFT
lap_spectral = real(ifft2(lap_hat));

%% 5. Error Analysis
error = abs(lap_spectral - lap_exact);
max_error = max(error(:));

fprintf('Grid: %d x %d\n', Nx, Ny);
fprintf('Max Error: %.4e\n', max_error);