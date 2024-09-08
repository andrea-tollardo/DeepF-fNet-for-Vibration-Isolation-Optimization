close all
clear all
clc

%==========================================%
%                                          %
%   DeepF-fNet: tri-chiral honeycomb LRM   %
%                                          %
%       IEPS validation with Comsol        %
%                                          %
%         Author: Andrea Tollardo          %
%                                          %
%==========================================%

fprintf('Loading Comsol model ...\n');
model = mphload('Direct_Band-gap_Problem2D'); % Load Comsol model
fprintf('Comsol model successfully loaded\n');

% Load structural parameters
load(Params.mat);
theta = atan(2*r/L); %[rad] ligament angle of tri-chiral honeycomb

k = linspace(-1,1,40); %[-] Comsol sweep study parameter
n_cell_H = 5; % number of horizontal repetitions of hexagons
w_ALR = 3*L*n_cell_H/cos(theta); %[m] width of the ALR 
w_HS = 2.5*w_ALR; %[m] width of the HS
kx = k*pi/w_HS; %[rad/m] discretized wave number range

% Set new parameters for Comsol study
r_new = sprintf('%f [m]', r);
L_new = sprintf('%f [m]', L);
s_new = sprintf('%f [m]', s);
model.param.set('r',r_new);
model.param.set('L',L_new);
model.param.set('s',s_new);

% Start Comsol sweep study
fprintf('Starting sweep study ...\n');
    for uu=1:length(k) % sweep all the wave number range
        k_new = sprintf('%f',k(uu)); % set wave number for Comsol study
        model.param.set('k',k_new);
        model.study('std1').run; % run Comsol study
        fq(:,uu) = real(mphglobal(model,'freq')); %[Hz] dispersion curve
    end

fprintf('Sweep study successfully completed\n');
fprintf('Saving results ...\n')
omega2 = (2*pi*fq).^2; %[rad2/s2] conversion to SI units
save('omega2Comsol','omega2','kx'); % save dispersion curve to .mat file
fprintf('Results successfully saved\n');
