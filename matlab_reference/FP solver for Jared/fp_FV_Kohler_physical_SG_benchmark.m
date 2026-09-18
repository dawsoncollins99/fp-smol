% fp_FV_Kohler_physical_SG

% Steady FV solution of 1D Fokker-Planck in x = r^2, with:
% - Drift in x: A_x(x)
% - Diffusion Ds
% - Settling loss lambda(x) = (C/h)*x
% Includes:
%   (1) Option for constant drift A_x = a*s  (Kohler OFF)
%   (2) Option for Kohler drift A_x = a (s - A_k/r + B_k/r^3) (Kohler ON)
% Always compares FV result to Airy analytic solution for the
% constant-drift case.


% [x]  Compared with general (Airy) solutions obtained previously and 
%      plotted. There is overall good agreement, but some differences for 
%      SS =< 0. But FV solution matches calculated Airy solution.

% [x] look at solutions with small K (D_x) with Kohler terms.
%     Seems like I can get to limit for K = 0 with K = 1e-4 um^4/s
% [ ] Add SS, K and r_m to each plot; remove Airy baseline.
% [ ] Plot vs log r.

% [x] Replaced advection with SG scheme; slightly better for bimodal DSDs.
% [x] Added scaled Airy to prevent NaNs.

% [x] 11-30-25 -- corrected calculation of mean radius and of radius pdf.

% [x] 2-24-26 -- compare to Grant's for UGN; change r_dry (7, 25), s (-.05), 
%                D_x (small), r_max (70 for r_dry=25); modified figure 6

% [x] 3-2-26 -- cleaned up.
% [x] 3-3-26 -- plot pdf(r) not n(r).

% [x] 3-4-26 -- for r_dry   = 1.3 um  r^2_mode = 91 for s >= - 0.003.
%      As r_dry decreases, min s for a solution increases, and r^2_mode
%      decreases. For r_dry = 0.163 um, s_min = 0.05%, r^2_mode = 10.4

clear; 
% clc;

%% --------------------------
% PARAMETERS (dimensionless)
%% --------------------------

%% --- Physical parameters (SI) ---

um_per_m = 1.e6;

% Thermodynamic / fluid properties
T       = 288.0;           % K
p       = 1000e2;          % Pa
rgas    = 287;             % J/kg/K
rho_air = p / (rgas * T);  % kg/m^3
rho_w   = 1000.0;          % kg/m^3
g       = 9.81;            % m/s^2
mu_air  = 1.8e-5;          % Pa*s
sigma_v = 0.075;           % N/m
M_w     = 18e-3;           % kg/mol
R       = 8.314;           % J/(mol*K)


% Solute: NaCl example
rho_s   = 2160.0;          % kg/m^3
M_s     = 58.5e-3;         % kg/mol
nu_s    = 2.0;             % van 't Hoff factor

r_dry   = 0.065e-6;                               % m, dry radius
% r_dry   = 7e-6;                               % m, dry radius 
% r_dry   =  0.5 * 7e-6;   % corey diam=7
% r_dry   =  0.5 * 5e-6;   % corey diam=5

% r_dry   = 0.8 * 7e-6;  % 0.3-0.9 produces an interesting sequence for s = - 0.0114.  

% r_dry   = 20 * 0.065e-6;  
% r_dry   = 10 * 0.065e-6;  
% r_dry   = 5 * 0.065e-6;  
% r_dry   = 2.5 * 0.065e-6;  

m_s     = (4/3)*pi * r_dry.^3 * rho_s;           % kg solute per droplet

% Koeler parameters A [m], B [m^3]
A_k = 2 * sigma_v * M_w / (rho_w * R * T);       % curvature
B_k = 3 * nu_s * m_s * M_w / (4*pi * rho_w * M_s);  % solute

um_per_m = 1e6;
A_k_um = A_k * um_per_m;
B_k_um3 = B_k * um_per_m^3;

% Supersaturation and growth
% s       = 0.0100;           % dimensionless supersaturation (1%)
% s       = - 0.05; 
% s       = - 0.0114; 
% s       = - 0.10; 
% s       = - 0.003;    % smallest s for solution with r_dry   = 20 * 0.065e-6;  
% s       = - 0.01;
% s       = - 0.02;
% s       = - 0.04;
% s       =   0.00;
% s       =   0.001/2;
% s       =   0.001;
% s       =   0.01;
% s       =   0.02;
% s       =   0.04;   % solution exists with r_dry   = 20 * 0.065e-6; 

s       = 0.0009;


a_phys = 2 * xi1_func(T,p) / um_per_m^2; % m^2/s, dr^2/dt = a_phys*(s - s_eq) 

% Supersaturation diffusivity in x = r^2
D_x     = 1.0e-16;          % m^4/s  (this plays the role of your old Ds)
% D_x     = D_x / 1e10;     % D_x / 1e10 = 0.01 um^4/s
D_x = D_x / 1e7;            % gives D_x = 10 um^4/s
% D_x     = D_x / 1e14;     % essentially no ss fluctuations

D_x_um4 = D_x * um_per_m^4;

% Chamber geometry and settling
h       = 1.0;           % m

C_stk   = 2*(rho_w - rho_air)*g / (9*mu_air);   % [s^-1 m^-1]  1.2097e+08
k_1 = C_stk;

% Injection flux at left boundary in x-space [arbitrary]
Jinj    = 1.0e8;          % number/(m^2 s) (overall scale factor)

% x-grid
Ncells  = 4000;
Ncells  = 8000;
r_min   = 0.001e-6; % m
r_min = r_dry/2;  % for r_min < r_dry, actual value does not matter

r_max   = 30e-6;          % 
% r_max   = 60e-6;
% r_max   = 70e-6;         % m           rdry = 25

x_min   = r_min^2;  
x_max   = r_max^2;

r_max_um = 15; % for plotting

% use_kohler = false;  % <-- set to true to turn on Kohler drift
use_kohler = true

s_pct = s * 100

if use_kohler
    r_crit_um = sqrt(3 * B_k / A_k) * um_per_m
    s_crit_pct = sqrt(4 * A_k^3 / (27 * B_k)) * 100
end

if ~use_kohler
    r_min   = 0;
    x_min   = r_min^2;
end

% NO RUN PARAMETERS TO SET IN CODE BELOW

%% --------------------------
% GRID
%% --------------------------

xf = linspace(x_min, x_max, Ncells+1);           % faces
xc = 0.5*(xf(1:end-1) + xf(2:end));             % centers
dx = xf(2) - xf(1);

%% --------------------------
% DRIFT A_x(x)
%% --------------------------

if ~use_kohler
    Ax0   = a_phys * s;           % constant drift [m^2/s]
    % Constant drift case (baseline)
    Ax_c    = Ax0 * ones(size(xc));            % center drift
    Ax_face = Ax0 * ones(1, Ncells-1);         % face drift
else
    % Kohler drift: A_x(x) = a [ s - A_k/r + B_k/r^3 ], r = sqrt(x)

    %% --- Drift and loss in physical units ---

    % radii at cell centers [m]
    r_c = sqrt(xc);                % xc is [m^2]

    % equilibrium supersaturation s_eq(r) = A/r - B/r^3
    s_eq = A_k ./ r_c - B_k ./ (r_c.^3);

    % drift in x = r^2: A_x(x) = a_phys * (s - s_eq)
    Ax_c = a_phys * (s - s_eq);            % [m^2/s] at centers

    % drift at faces (simple average)
    Ax_face = 0.5 * (Ax_c(1:end-1) + Ax_c(2:end));   % length Ncells-1
end

% --- Scharfetter-Gummel coefficients at faces ---

Pe_face = Ax_face * dx / D_x;    % Peclet number at each face (dimensionless)
alpha_face = zeros(1, Ncells-1);
beta_face  = zeros(1, Ncells-1);

for j = 1:Ncells-1
    Pe = Pe_face(j);
    % Bernoulli function B(z) = z / (exp(z) - 1), with small-z fix
    if abs(Pe) < 1.0e-3
        % Taylor expansion: B(z) ~ 1 - z/2 + z^2/12
        Bp  = 1 - Pe/2 + (Pe^2)/12;
        Bm  = 1 + Pe/2 + (Pe^2)/12;   % B(-Pe)
    else
        Bp  = Pe/(exp(Pe) - 1);       % B(Pe)
        Bm  = -Pe/(exp(-Pe) - 1);     % B(-Pe) = -Pe / (exp(-Pe) - 1)
    end

    alpha_face(j) = (D_x/dx) * Bm;    % = (D_x/dx)*B(-Pe)
    beta_face(j)  = -(D_x/dx) * Bp;   % = -(D_x/dx)*B(Pe)
end

% settling loss at centers: lambda(x) = (C_stk/h)*x
lambda = (C_stk / h) * xc;             % [1/s]

%% --------------------------
% ASSEMBLE TRIDIAGONAL SYSTEM A * N = b
%% --------------------------

A = zeros(Ncells, Ncells);
b = zeros(Ncells, 1);

% Row 1: J_{3/2} - J_{1/2} + lambda_1 N_1 dx = 0, with J_{1/2} = Jinj
A(1,1) = lambda(1)*dx + alpha_face(1);
A(1,2) = beta_face(1);
b(1)   = Jinj;

% Interior rows: i = 2..Ncells-1
for i = 2:Ncells-1
    A(i,i-1) = -alpha_face(i-1);
    A(i,i)   =  alpha_face(i) - beta_face(i-1) + lambda(i)*dx;
    A(i,i+1) =  beta_face(i);
end

% Last row: -J_{N-1/2} + lambda_N N_N dx = 0, J_{N+1/2} = 0
A(Ncells, Ncells-1) = -alpha_face(Ncells-1);
A(Ncells, Ncells)   = -beta_face(Ncells-1) + lambda(Ncells)*dx;

%% --------------------------
% SOLVE FV SYSTEM
% --------------------------

N_fv = A \ b;

%% --------------------------
% ANALYSIS AND PLOTTING CODE
% --------------------------

% PDF-in-r computation--THIS IS IMPORTANT TO GET PDFs CORRECT.
% The code above calculates the PSD of x = r^2, in bins equally spaced in x.
% --------------------------

% Column vectors (avoid row/col mismatches)
xc   = xc(:);              % [m^2], uniform
N_fv = N_fv(:);            % FV state

% Radius
r = sqrt(xc);              % [m]

%    Bin counts with respect to r (change of variables x = r^2)
% CORRECT FOR FV SOLVER
n_r = 2 .* r .* N_fv;      % [m^-1]  = dN/dr

%    Mean radius 
% CORRECT FOR FV SOLVER
r_m_um = sum( sqrt(xc) .* N_fv ) / sum(N_fv) * um_per_m

%    Pdf in radius

% Normalize with respect to r using trapz, NOT sum:
Z = trapz(r, n_r);           % integral over r
pdf_r = n_r / Z;             % now trapz(r, pdf_r) ≈ 1

% not needed but gives same answer
r_mean_pdf = trapz(r, r .* pdf_r)  * um_per_m   % consistent with everything else

%    Optional check
% CORRECT FOR FV SOLVER
check = trapz(r, pdf_r);   % should be ~ 1

fprintf('check = %.15g\n', check)

% PDF in x = r^2

Zx = trapz(xc, N_fv);     % total number
pdf_x = N_fv / Zx;        % probability density in x

% find mode of pdf_x

x = xc;

[pmax,k] = max(pdf_x);
if k>1 && k<numel(x)
    dx = x(2)-x(1);
    y1=pdf_x(k-1); y2=pdf_x(k); y3=pdf_x(k+1);
    x_mode = x(k) + 0.5*dx*(y1 - y3)/(y1 - 2*y2 + y3);
else
    x_mode = x(k);
end

fprintf('r^2_mode = %.6g\n', x_mode * 1e12)


%% --------------------------
% PLOTS
%% --------------------------

if 1==1

    % 3. Radius distributions n(r)

    figure(25);
    clf
%     plot(r*1e6, n_r, '-','linewidth',2);
    plot(r*1e6, pdf_r, '-','linewidth',2);
    xlabel('r ({\mu}m)');
%     ylabel('n(r) (m^{-1})');
    ylabel('p_r(r) (m^{-1})')
    if use_kohler
%         legend('FV + SG with Koeler', 'Location', 'best');
        title('pdf(r): FV + SG with Kohler drift');
    else
%         legend('FV + SG without Koeler', 'Location', 'best');
        title('pdf(r): FV + SG constant drift');
    end
    grid on;
    set(gca,'LineWidth',1.5,'FontSize',14)

    xlim([0 r_max_um])

    txt = {
    sprintf('s (%%) = %.3g', s_pct)
    sprintf('D\\_x (um^4/s) = %.3g', D_x_um4)
    sprintf('a\\_phys (um^2/s) = %.3g', a_phys*1e12)
    sprintf('r\\_m (um) = %.3g', r_m_um)
    sprintf('r_{dry} (um) = %.3g', r_dry*1e6)
};

text(0.65, 0.85, txt, ...
    'Units','normalized', ...
    'VerticalAlignment','top', ...
    'FontSize',14, ...
    'BackgroundColor',[1 1 1]*0.95, ...  % light gray box
    'Margin',6);

end

    fn = sprintf('s%.3g_Dx%.3g_a%.3g_rdry%.3g_pr_r.pdf', ...
            s_pct, D_x_um4, a_phys*1e12, r_dry*1e6);
     exportgraphics(gcf, fn, 'ContentType','vector');


if 1==2

    % 3. Radius distributions n(r)

    figure(6);
    clf
%     semilogy(r*1e6, n_r, '-', 'linewidth',2);
    semilogy(r*1e6, pdf_r, '-', 'linewidth',2);
    xlabel('r ({\mu}m)');
    ylabel('p_r(r) (m^{-1})')
    if use_kohler
%         legend('FV + SG with Koeler', 'Location', 'northwest');
        title('pdf(r): FV + SG with Kohler drift');
    else
%         legend('FV + SG without Koeler', 'Location', 'northwest');
        title('pdf(r): FV + SG constant drift vs Airy baseline');
    end
    grid on;

    set(gca,'LineWidth',1.5,'FontSize',14)
    text(0.05, 0.85, txt, ...
    'Units','normalized', ...
    'VerticalAlignment','top', ...
    'FontSize',14, ...
    'BackgroundColor',[1 1 1]*0.95, ...  % light gray box
    'Margin',6);

    fn = sprintf('s%.3g_Dx%.3g_a%.3g_rdry%.3g_pr_r_logy.pdf', ...
            s_pct, D_x_um4, a_phys*1e12, r_dry*1e6);
     exportgraphics(gcf, fn, 'ContentType','vector');

end

%%

figure(3)
plot(xc*1e6*1e6, pdf_x,'LineWidth',1.5)
xlabel('x = r^2 ({\mu}m^2)')
ylabel('p_x(x) (m^{-2})')
set(gca,'LineWidth',1.5,'FontSize',16)

 grid on;

 xlim([0 r_max_um^2])
 ylim([0 1e10])

     txt = {
    sprintf('s (%%) = %.3g', s_pct)
    sprintf('D\\_x (um^4/s) = %.3g', D_x_um4)
    sprintf('a\\_phys (um^2/s) = %.3g', a_phys*1e12)
    sprintf('r\\_m (um) = %.3g', r_m_um)
    sprintf('r^2_{mode} (um^2) = %.3g', x_mode * 1e12)
    sprintf('r_{dry} (um) = %.3g', r_dry*1e6)
};

text(0.65, 0.85, txt, ...
    'Units','normalized', ...
    'VerticalAlignment','top', ...
    'FontSize',14, ...
    'BackgroundColor',[1 1 1]*0.95, ...  % light gray box
    'Margin',6);

    fn = sprintf('s%.3g_Dx%.3g_a%.3g_rdry%.3g_px_x.pdf', ...
            s_pct, D_x_um4, a_phys*1e12, r_dry*1e6);
     exportgraphics(gcf, fn, 'ContentType','vector');
