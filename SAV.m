% =========================================================================
% SAV Method for Cahn-Hilliard Equation (2D)
% Scheme: 1st order SAV-BDF1
% Ref: Shen, J., Xu, J., & Yang, J. (2018). JCP.
% =========================================================================

clear; close all; clc;

%% 1. Parameters & Grid
Lx = 2*pi; Ly = 2*pi;
Nx = 128;  Ny = 128;
dt = 1e-3;   % Time step
T_end = 1.0; 
n_steps = ceil(T_end/dt);

M = 1.0; epsilon = 0.05; C0 = 0.0; % C0 ensures sqrt>0

% Grid and Spectral Operators
dx = Lx/Nx; dy = Ly/Ny;
x = dx*(0:Nx-1)'; y = dy*(0:Ny-1)';
[X, Y] = ndgrid(x, y);

kx = [0:Nx/2-1, -Nx/2:-1]';
ky = [0:Ny/2-1, -Ny/2:-1]';
[Kx, Ky] = ndgrid(kx, ky);

k2 = Kx.^2 + Ky.^2; % -Laplacian
k4 = k2.^2;         % Bi-Laplacian

%% 2. Initial Condition
phi = 0.05 * (2*rand(Nx, Ny) - 1);
phi_hat = fft2(phi);

% Initial Energy and SAV variable r
F_phi = 0.25 * (phi.^2 - 1).^2;
E1 = sum(sum(F_phi)) * dx * dy;
r = sqrt(E1 + C0);

%% 3. Operators for Linear System
% The linear operator for CH: A = I + dt*M*eps^2*Delta^2
% In spectral space: A_hat = 1 + dt * M * epsilon^2 * k4
A_hat = 1 + dt * M * epsilon^2 * k4;

%% 4. Video Writer Initialization 
video_filename = 'Cahn_Hilliard_SAV_Evolution.mp4';
v = VideoWriter(video_filename, 'MPEG-4');
v.FrameRate = 10; % 设置帧率，可以根据需要调整流畅度
open(v);
fprintf('正在初始化视频录制... 输出文件: %s\n', video_filename);
%% 5. Time Evolution Loop
energy_hist = zeros(n_steps, 1);
h_fig = figure('Color','w', 'Position', [100, 100, 900, 400]); %以此句柄抓图

for n = 1:n_steps

    % --- Step 1: Pre-calculate Explicit Terms ---
    phi_n = real(ifft2(phi_hat));

    % Nonlinear derivative component: b = F'(phi_n) / sqrt(E1 + C0)
    F_prime = phi_n.^3 - phi_n;
    E1_n = sum(sum( 0.25*(phi_n.^2 - 1).^2 )) * dx * dy;
    sqrt_E = sqrt(E1_n + C0);
    b = F_prime / sqrt_E;  
    b_hat = fft2(b);

    % --- Step 2: Solve the Decoupled System (The SAV Strategy) ---
    % We solve via block elimination (Sherman-Morrison type approach)
    % System: A*phi_new + g*r_new = source

    % Define vector g corresponding to -dt*M*Laplace(b)
    g_hat = dt * M * k2 .* b_hat;

    % 1. Solve for intermediate variable psi: A * psi = phi^n
    psi_hat = phi_hat ./ A_hat;

    % 2. Solve for intermediate variable eta: A * eta = g
    eta_hat = g_hat ./ A_hat;

    % 3. Calculate inner products for the algebraic scalar equation
    % Integral terms: <b, psi> and <b, eta>
    b_delta_psi = sum(sum( b .* (real(ifft2(psi_hat)) - phi_n) )) * dx * dy;
    % b_psi = sum(sum( b .* real(ifft2(psi_hat)) )) * dx * dy;
    b_eta = sum(sum( b .* real(ifft2(eta_hat)) )) * dx * dy;

    % 4. Compute r^{n+1} explicitly
    % Formula derived from the scalar equation
    r_new = (r + 0.5 * b_delta_psi) / (1 + 0.5 * b_eta);

    % 5. Update phi^{n+1}
    phi_hat_new = psi_hat - r_new * eta_hat;

    % --- Update Variables ---
    phi_hat = phi_hat_new;
    r = r_new;

    % --- Step 3: Energy Verification ---
    % Note: SAV preserves the "Modified Energy"
    [dpx, dpy] = gradient(real(ifft2(phi_hat)), dx, dy);
    grad_E = 0.5 * epsilon^2 * sum(sum(dpx.^2 + dpy.^2)) * dx * dy;
    E_sav = grad_E + r^2;
    energy_hist(n) = E_sav;

    % --- Visualization ---
    if mod(n, 100) == 0
        subplot(1, 2, 1);
        imagesc(x, y, real(ifft2(phi_hat))'); 
        axis xy equal tight; axis off;
        colormap(jet); colorbar; clim([-1,1]);
        title(['SAV Method (t = ' num2str(n*dt) ')']);

        subplot(1, 2, 2);
        plot(1:n, energy_hist(1:n), 'r-', 'LineWidth', 1.5);
        title('Modified Energy Decay');
        grid on; xlim([0 n_steps]);
        drawnow;
   %捕捉并写入视频帧 
        frame = getframe(h_fig);
        writeVideo(v, frame);
    end
end

%% 6. Finalize Video
close(v);
fprintf('模拟结束。视频保存成功！文件名：%s\n', video_filename);