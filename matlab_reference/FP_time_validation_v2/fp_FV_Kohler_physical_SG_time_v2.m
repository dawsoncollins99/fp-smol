% fp_FV_Kohler_physical_SG_time_v2
%
% Time-dependent finite-volume solution of the 1-D Fokker-Planck equation
% in x = r^2, based directly on fp_FV_Kohler_physical_SG_benchmark.
%
% This v1 intentionally contains NO collision/coalescence.  Its purpose is
% to validate the time-dependent formulation by showing that it approaches
% the previously computed steady finite-volume solution.
%
% Governing cell balance:
%
%   dN_i/dt = -(J_{i+1/2}-J_{i-1/2})/dx - lambda_i N_i,
%
% with prescribed injection flux Jinj at the left boundary and zero FP flux
% at the right boundary.  The steady discretization can be written
%
%   A*N = b,
%
% so the time-dependent semi-discrete system is simply
%
%   dN/dt = (b - A*N)/dx.
%
% Backward Euler gives
%
%   [I + (dt/dx) A] N^{n+1} = N^n + (dt/dx) b.
%
% The Scharfetter-Gummel spatial discretization, Kohler drift, settling loss,
% and finite-radius lower boundary are retained from the steady solver.
%
% Required helper functions in the MATLAB path:
%   xi1_func.m
%   esat_Pa.m
%
% 2026-08-21: time-dependent validation version v2.

clear
close all

%% ------------------------------------------------------------------------
% USER CONTROLS
% -------------------------------------------------------------------------

% Thermodynamic state
T = 288.0;                 % K
p = 1000e2;                % Pa

% Solute: NaCl
r_dry = 0.065e-6;          % m

% Ambient supersaturation
s = 0.0009;                % dimensionless (0.09%)

% Stochastic-condensation diffusivity in x = r^2.
% Specify it directly in um^4/s, then convert once to SI.
D_x_um4 = 1.0e-6;          % um^4/s
D_x_um4 = 1.0e1;          % um^4/s

% Chamber / fallout
h = 1.0;                   % m

% Injection flux through the left x-face
Jinj = 1.0e8;              % number/(m^2 s)

% x = r^2 grid
Ncells = 8000;
r_min = r_dry/2;           % m; finite lower radius avoids r=0 Kohler issue
r_max = 30e-6;             % m

% Kohler drift switch
use_kohler = true;

% Time integration
% Backward Euler is unconditionally stable for the linear FP step, but dt
% still controls temporal accuracy.
dt = 1.0;                  % s
t_end = 2000.0;            % s

% Times to retain for DSD plots. Values outside [0,t_end] are ignored.
snapshot_times_requested = [10 50 200 500 1000 1500 2000];  % s

% Optional figure export
export_plots = false;

%% ------------------------------------------------------------------------
% PHYSICAL PARAMETERS
% -------------------------------------------------------------------------

um_per_m = 1.e6;

rgas    = 287;             % J/(kg K)
rho_air = p/(rgas*T);      % kg/m^3
rho_w   = 1000.0;          % kg/m^3
g       = 9.81;            % m/s^2
mu_air  = 1.8e-5;          % Pa s
sigma_v = 0.075;           % N/m
M_w     = 18e-3;           % kg/mol
R       = 8.314;            % J/(mol K)

rho_s = 2160.0;            % kg/m^3, NaCl
M_s   = 58.5e-3;           % kg/mol
nu_s  = 2.0;               % van 't Hoff factor

m_s = (4/3)*pi*r_dry^3*rho_s;

% Kohler parameters: s_eq = A_k/r - B_k/r^3
A_k = 2*sigma_v*M_w/(rho_w*R*T);                         % m
B_k = 3*nu_s*m_s*M_w/(4*pi*rho_w*M_s);                  % m^3

% xi1_func returns um^2/s.  Convert 2*xi1 to m^2/s for d(r^2)/dt.
a_phys = 2*xi1_func(T,p)/um_per_m^2;                     % m^2/s

% Convert D_x from um^4/s to m^4/s.
D_x = D_x_um4/um_per_m^4;                                % m^4/s

% Stokes settling coefficient: v_t = C_stk r^2.
C_stk = 2*(rho_w-rho_air)*g/(9*mu_air);                  % 1/(m s)

s_pct = 100*s;

if use_kohler
    r_crit_um = sqrt(3*B_k/A_k)*um_per_m;
    s_crit_pct = sqrt(4*A_k^3/(27*B_k))*100;
else
    r_min = 0;
    r_crit_um = NaN;
    s_crit_pct = NaN;
end

%% ------------------------------------------------------------------------
% GRID
% -------------------------------------------------------------------------

x_min = r_min^2;
x_max = r_max^2;

xf = linspace(x_min,x_max,Ncells+1);      % cell faces [m^2]
xc = 0.5*(xf(1:end-1)+xf(2:end));         % cell centers [m^2]
dx = xf(2)-xf(1);                         % uniform spacing [m^2]

xc = xc(:);
r = sqrt(xc);                             % center radii [m]

%% ------------------------------------------------------------------------
% DRIFT A_x(x)
% -------------------------------------------------------------------------

if ~use_kohler
    Ax0 = a_phys*s;
    Ax_c = Ax0*ones(Ncells,1);
else
    % Kohler equilibrium supersaturation at cell centers.
    s_eq = A_k./r - B_k./(r.^3);
    Ax_c = a_phys*(s-s_eq);                % m^2/s
end

% Interior-face drift by arithmetic averaging.
Ax_face = 0.5*(Ax_c(1:end-1)+Ax_c(2:end));

%% ------------------------------------------------------------------------
% SCHARFETTER-GUMMEL FLUX COEFFICIENTS
% -------------------------------------------------------------------------
%
% At an interior face,
%
%   J_{i+1/2} = alpha_i*N_i + beta_i*N_{i+1}.
%
% The large-|Pe| branches below avoid overflow in exp(Pe) for very small D_x.

Pe_face = Ax_face*dx/D_x;
alpha_face = zeros(Ncells-1,1);
beta_face  = zeros(Ncells-1,1);

for j = 1:Ncells-1
    Pe = Pe_face(j);

    if abs(Pe) < 1.0e-3
        % B(Pe) and B(-Pe), using Taylor series around zero.
        Bp = 1 - Pe/2 + Pe^2/12;
        Bm = 1 + Pe/2 + Pe^2/12;
    elseif Pe > 50
        % B(Pe) -> 0, B(-Pe) -> Pe.
        Bp = 0;
        Bm = Pe;
    elseif Pe < -50
        % B(Pe) -> -Pe, B(-Pe) -> 0.
        Bp = -Pe;
        Bm = 0;
    else
        % expm1 improves accuracy when Pe is moderately small.
        Bp = Pe/expm1(Pe);
        Bm = (-Pe)/expm1(-Pe);
    end

    alpha_face(j) = (D_x/dx)*Bm;
    beta_face(j)  = -(D_x/dx)*Bp;
end

%% ------------------------------------------------------------------------
% SETTLING LOSS
% -------------------------------------------------------------------------

lambda = (C_stk/h)*xc;                    % 1/s

%% ------------------------------------------------------------------------
% ASSEMBLE THE SAME STEADY FV OPERATOR A*N = b, BUT SPARSELY
% -------------------------------------------------------------------------

main_diag = zeros(Ncells,1);
lower_diag = zeros(Ncells,1);              % row i, column i-1; i >= 2
upper_diag = zeros(Ncells,1);              % row i, column i+1; i <= N-1

% First row: J_{3/2} - Jinj + lambda_1*N_1*dx = 0.
main_diag(1) = lambda(1)*dx + alpha_face(1);
upper_diag(1) = beta_face(1);

% Interior rows.
for i = 2:Ncells-1
    lower_diag(i) = -alpha_face(i-1);
    main_diag(i)  = alpha_face(i) - beta_face(i-1) + lambda(i)*dx;
    upper_diag(i) = beta_face(i);
end

% Last row: -J_{N-1/2} + lambda_N*N_N*dx = 0,
% with J_{N+1/2}=0.
lower_diag(Ncells) = -alpha_face(Ncells-1);
main_diag(Ncells)  = -beta_face(Ncells-1) + lambda(Ncells)*dx;

rows = [(1:Ncells)'; (2:Ncells)'; (1:Ncells-1)'];
cols = [(1:Ncells)'; (1:Ncells-1)'; (2:Ncells)'];
vals = [main_diag; lower_diag(2:end); upper_diag(1:end-1)];
A = sparse(rows,cols,vals,Ncells,Ncells);

b = zeros(Ncells,1);
b(1) = Jinj;

%% ------------------------------------------------------------------------
% REFERENCE STEADY SOLUTION
% -------------------------------------------------------------------------

N_ss = A\b;

if any(~isfinite(N_ss)) || any(N_ss < 0)
    warning('Reference steady solution contains nonfinite or negative values.')
end

Ntot_ss = sum(N_ss)*dx;                    % number/m^3 air
rmean_ss_um = sum(r.*N_ss)/sum(N_ss)*um_per_m;
steady_residual = norm(A*N_ss-b,1)/max(norm(b,1),realmin);

%% ------------------------------------------------------------------------
% TIME INTEGRATION: BACKWARD EULER
% -------------------------------------------------------------------------

nsteps = round(t_end/dt);
t_end_actual = nsteps*dt;
time = (0:nsteps)'*dt;

% dN/dt = (b-A*N)/dx
% => [I + dt*A/dx] N^{n+1} = N^n + dt*b/dx.
M = speye(Ncells) + (dt/dx)*A;
source_increment = (dt/dx)*b;

% Factor the constant matrix once when MATLAB's decomposition object exists.
use_decomposition = (exist('decomposition','builtin') == 5) || ...
                    (exist('decomposition','file') == 2);
if use_decomposition
    Msolve = decomposition(M,'lu');
end

% Empty initial chamber is a stringent test of approach to the old steady
% solution.  This can later be replaced by any desired initial DSD.
N = zeros(Ncells,1);

rel_L1 = zeros(nsteps+1,1);
Ntot = zeros(nsteps+1,1);
rmean_um = nan(nsteps+1,1);

rel_L1(1) = sum(abs(N-N_ss))/sum(abs(N_ss));
Ntot(1) = sum(N)*dx;

% Prepare requested snapshots.
snapshot_times = snapshot_times_requested(:);
snapshot_times = snapshot_times(snapshot_times >= 0 & ...
                                snapshot_times <= t_end_actual);
snapshot_steps = round(snapshot_times/dt);
[snapshot_steps,ia] = unique(snapshot_steps,'stable');
snapshot_times = snapshot_times(ia);
Nsnap = nan(Ncells,numel(snapshot_steps));
snapshot_saved = false(size(snapshot_steps));

% Save t=0 if explicitly requested.
for is = 1:numel(snapshot_steps)
    if snapshot_steps(is) == 0
        Nsnap(:,is) = N;
        snapshot_saved(is) = true;
    end
end

fprintf('\nTIME-DEPENDENT FP VALIDATION (NO COALESCENCE)\n')
fprintf('Ncells                  = %d\n',Ncells)
fprintf('dt, t_end               = %.6g, %.6g s\n',dt,t_end_actual)
fprintf('s                       = %.6g %%\n',s_pct)
fprintf('D_x                     = %.6g um^4/s\n',D_x_um4)
fprintf('r_dry                   = %.6g um\n',r_dry*um_per_m)
fprintf('steady mean radius      = %.6g um\n',rmean_ss_um)
fprintf('steady linear residual  = %.3e\n',steady_residual)
if use_kohler
    fprintf('Kohler r_crit, s_crit   = %.6g um, %.6g %%\n', ...
            r_crit_um,s_crit_pct)
end

print_every = max(1,round(nsteps/10));

for k = 1:nsteps
    rhs = N + source_increment;

    if use_decomposition
        Nnew = Msolve\rhs;
    else
        Nnew = M\rhs;
    end

    % SG + backward Euler should preserve positivity.  Remove only tiny
    % roundoff negatives, and warn if a significant negative appears.
    scaleN = max(max(N_ss),realmin);
    neg_tol = 1.0e-12*scaleN;
    if any(Nnew < -neg_tol)
        warning('Significant negative N encountered at t = %.6g s.',k*dt)
    end
    tiny_negative = (Nnew < 0) & (Nnew >= -neg_tol);
    Nnew(tiny_negative) = 0;

    N = Nnew;

    rel_L1(k+1) = sum(abs(N-N_ss))/sum(abs(N_ss));
    Ntot(k+1) = sum(N)*dx;
    if sum(N) > 0
        rmean_um(k+1) = sum(r.*N)/sum(N)*um_per_m;
    end

    for is = 1:numel(snapshot_steps)
        if ~snapshot_saved(is) && k == snapshot_steps(is)
            Nsnap(:,is) = N;
            snapshot_saved(is) = true;
        end
    end

    if mod(k,print_every) == 0 || k == nsteps
        fprintf('t = %8.1f s : rel L1 to steady = %.6e, N/Nss = %.6f\n', ...
                k*dt,rel_L1(k+1),Ntot(k+1)/Ntot_ss)
    end
end

%% ------------------------------------------------------------------------
% FINAL VALIDATION DIAGNOSTICS
% -------------------------------------------------------------------------

final_rel_L1 = rel_L1(end);
final_N_ratio = Ntot(end)/Ntot_ss;
final_rmean_um = rmean_um(end);

fprintf('\nFINAL VALIDATION\n')
fprintf('relative L1 error to steady N(x) = %.8e\n',final_rel_L1)
fprintf('total-number ratio N(t)/N_ss     = %.8f\n',final_N_ratio)
fprintf('mean radius at final time        = %.8f um\n',final_rmean_um)
fprintf('steady mean radius               = %.8f um\n',rmean_ss_um)

%% ------------------------------------------------------------------------
% FIGURE 1: EVOLUTION OF THE RADIUS PDF TOWARD THE STEADY PDF
% -------------------------------------------------------------------------

figure(1)
clf
hold on

for is = 1:numel(snapshot_steps)
    if ~snapshot_saved(is)
        continue
    end

    Nplot = Nsnap(:,is);
    n_r = 2*r.*Nplot;                       % dN/dr
    Zr = trapz(r,n_r);

    if Zr > 0
        pdf_r = n_r/Zr;
        plot(r*um_per_m,pdf_r,'LineWidth',1.4, ...
             'DisplayName',sprintf('t = %.0f s',snapshot_steps(is)*dt))
    end
end

n_r_ss = 2*r.*N_ss;
Zr_ss = trapz(r,n_r_ss);
pdf_r_ss = n_r_ss/Zr_ss;
plot(r*um_per_m,pdf_r_ss,'--','LineWidth',2.0, ...
     'DisplayName','steady FV')

xlabel('r ({\mu}m)')
ylabel('p_r(r) (m^{-1})')
title('Time-dependent FP solution approaching steady FV solution')
legend('Location','best')
grid on
set(gca,'LineWidth',1.5,'FontSize',14)
xlim([0 r_max*um_per_m])
hold off

%% ------------------------------------------------------------------------
% FIGURE 2: QUANTITATIVE CONVERGENCE TO THE OLD STEADY SOLUTION
% -------------------------------------------------------------------------

figure(2)
clf
semilogy(time,rel_L1,'LineWidth',1.6)
xlabel('time (s)')
ylabel('relative L_1 error to steady N(x)')
title('Convergence of time-dependent FP solution')
grid on
set(gca,'LineWidth',1.5,'FontSize',14)

%% ------------------------------------------------------------------------
% FIGURE 3: TOTAL NUMBER AND MEAN RADIUS VS TIME
% -------------------------------------------------------------------------

figure(3)
clf

tiledlayout(2,1,'TileSpacing','compact','Padding','compact')

ax1 = nexttile;
plot(time,Ntot/Ntot_ss,'LineWidth',1.6)
yline(1,'--','LineWidth',1.2)
ylabel('N_{tot}(t) / N_{tot,ss}')
grid on
set(ax1,'LineWidth',1.5,'FontSize',14)

ax2 = nexttile;
plot(time,rmean_um,'LineWidth',1.6)
yline(rmean_ss_um,'--','LineWidth',1.2)
xlabel('time (s)')
ylabel('mean radius ({\mu}m)')
grid on
set(ax2,'LineWidth',1.5,'FontSize',14)

linkaxes([ax1 ax2],'x')

figure(4)

plot(r*um_per_m,pdf_r_ss,'--','LineWidth',2.0, ...
     'DisplayName','steady FV')

xlabel('r ({\mu}m)')
ylabel('p_r(r) (m^{-1})')
title('Time-dependent FP solution approaching steady FV solution')
legend('Location','best')
grid on
set(gca,'LineWidth',1.5,'FontSize',14)
xlim([0 15])
hold off


%% ------------------------------------------------------------------------
% OPTIONAL EXPORT
% -------------------------------------------------------------------------

if export_plots
    exportgraphics(figure(1),'FP_time_validation_pdf_r_v1.pdf', ...
                   'ContentType','vector')
    exportgraphics(figure(2),'FP_time_convergence_v1.pdf', ...
                   'ContentType','vector')
    exportgraphics(figure(3),'FP_time_number_mean_radius_v1.pdf', ...
                   'ContentType','vector')
end
