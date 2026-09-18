function [outputArg1] = xi1_func(T, press)
% xi1: dr_dt = xi1 * (S-1) / r
%   Input:
%       T (K)
%       press (Pa)
% functions: esat_Pa(T)
%   Output:
%     outputArg1 = xi1 (um^2/s)

um_per_m = 1e6;

rhow = 1.e3; % J/kg
T0 = 273.15;

Lv =  2.5e6; % J/kg  % latent of sublimation
Rv = 461.5;    % gas constant for water vapor

% -- Table 7.1 (p. 103): linear interpolation between -10 and 30 deg C
% -- Rogers and Yau (1989)
% (from fcnkb.f90)
K =  7.7e-5*(T-T0)+0.02399;   % thermal conductivity
% K = 0.0240  % 0 C

% www.igf.fuw.edu.pl
% K = 4.1868e-3 * (5.69 + 0.017 * (T - T0)) % W / (m K) 
% K = 0.0238; % (W / (m K) ) 0 C

% diffusivity of water vapor (from fcnkb.f90)
D = 1.57e-7*(T-273.15)+2.211e-5;
p0 = 1.e5;
D = D * p0 / press;

% www.igf.fuw.edu.pl
% p0 = 101325 % Pa
% D = 2.11e-5 * (T/T0)^1.94 * (p0/press);
% D = 2.11e-5; % (m^2/s) T=0 C, p = 1013.25 hPa

% accomodation coefficients = 1

denom = rhow * ...
( (Lv / (Rv * T)) * Lv / (K * T)  + Rv * T / ( esat_Pa(T) * D) );

outputArg1 = um_per_m^2 / denom; % (um^2/s)

end

