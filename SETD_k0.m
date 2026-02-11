% =========================================================================
% Stabilized ETD1 (SETD1) Solver for Allen-Cahn
% Problem: u_t = eps^2*Delta u + (u - u^3)
% Technique: Stabilized Linear Operator L = eps^2*Delta - kappa*I
% Author: Nabla @ Numerical Workshop
% =========================================================================

clear; close all; clc;

%% 1. Parameters
Lx = 2*pi; Ly = 2*pi;
Nx = 128;  Ny = 128;
dt = 0.05; T_end = 10.0; % 注意：这里用了较大的 dt
epsilon = 0.05; 
kappa = 0;            % 稳定化参数 (Stabilizer)

% Grid
dx = Lx/Nx; dy = Ly/Ny;
x = dx*(0:Nx-1)'; y = dy*(0:Ny-1)';
[X, Y] = ndgrid(x, y);

% Wave numbers
kx = [0:Nx/2-1, -Nx/2:-1]';
ky = [0:Ny/2-1, -Ny/2:-1]';
[Kx, Ky] = ndgrid(kx, ky);
k2 = Kx.^2 + Ky.^2;

%% 2. Initial Condition (Random Noise with Large Amplitude)
% 故意给一些接近边界的随机噪声，测试稳定性
u = 0.5 + 0.6 * (rand(Nx, Ny) - 0.5); % Range approx [0.2, 0.8]
u_hat = fft2(u);

%% 3. SETD Coefficients
% Modified Linear Operator: L_kappa = -eps^2 * k^2 - kappa
L_k = -epsilon^2 * k2 - kappa;

% Coefficients
E_hat = exp(L_k * dt);
Q_hat = (E_hat - 1) ./ L_k; % No singularity since L_k < 0

%% 4. Time Evolution
figure('Color','w', 'Position', [100, 100, 800, 400]);

max_val_hist = [];
min_val_hist = [];

for n = 1:ceil(T_end/dt)
    
    u_curr = real(ifft2(u_hat));
    
    % Check bounds
    max_val_hist(end+1) = max(u_curr(:));
    min_val_hist(end+1) = min(u_curr(:));
    
    % Modified Nonlinear Term: N_kappa = (u - u^3) + kappa*u
    Nu = (u_curr - u_curr.^3) + kappa * u_curr;
    Nu_hat = fft2(Nu);
    
    % Update
    u_hat = E_hat .* u_hat + Q_hat .* Nu_hat;
    
    % Visualization
    if mod(n, 10) == 0
        subplot(1, 2, 1);
        imagesc(x, y, real(ifft2(u_hat))');
        axis xy equal tight; axis off;
        colormap(jet); colorbar;
        title(['SETD1 (k=' num2str(kappa) ', t=' num2str(n*dt) ')']);
        
        subplot(1, 2, 2);
        plot(max_val_hist, 'r-', 'LineWidth', 1.5); hold on;
        plot(min_val_hist, 'b-', 'LineWidth', 1.5);
        yline(1, 'k--'); yline(-1, 'k--');
        title('Max/Min Value History');
        legend('Max', 'Min');
        xlim([0, n]); ylim([-1.2, 1.2]);
        drawnow;
    end
end