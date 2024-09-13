close all
clear all
clc

%==========================================%
%                                          %
%   DeepF-fNet: tri-chiral honeycomb LRM   %
%                                          %
%        WES validation with Comsol        %
%                                          %
%         Author: Andrea Tollardo          %
%                                          %
%==========================================%

% QUANTITIES TO CHANGE ACCORDING TO EIGENVECTOR
%==========================================
N = 1; % eigenfrequency index (1-10)
K = 1; % wave number (1-40)
p = 2; % select component: 1) u_Re 2) u_Im 3) v_Re 4) v_Im
%==========================================
load('Index.mat'); % load batch element index
i_rand = i_py + 1; % batch element index converted in MatLab
o = (N-1)*40+K-1; % ordered mode selected (0-399)

% Extract ground-truth displacement field
TT = load("TrainingTensorNN.mat","TrainingTensor"); % load the complete dataset
tt = TT.TrainingTensor(:,:,i_rand); % extract the batch element to compare
u_Re = real(tt(:,49+o*2)); %[m] horizontal displacement - real part
u_Im = imag(tt(:,49+o*2)); %[m] horizontal displacement - imaginary part
v_Re = real(tt(:,50+o*2)); %[m] vertical displacement - real part
v_Im = imag(tt(:,50+o*2)); %[m] vertical displacement - imaginary part
comp = [u_Re,u_Im,v_Re,v_Im]; % list of ground-truth eigenvector components

r = tt(2,1); %[m] ground-truth mean radius of tri-chiral honeycomb circle
L = tt(2,2); %[m] ground-truth ligament length of tri-chiral honeycomb
s = tt(2,3); %[m] ground-truth wall thickness of tri-chiral honeycomb
theta = atan(2*r/L); %[rad] ground-truth ligament angle of tri-chiral honeycomb
n_cell_H = 5; % number of horizontal repetitions of hexagons
n_cell_V = 9; % number of vertical repetitions of hexagons
h_ALR = sqrt(3)*L*n_cell_V/cos(theta); %[m] height of the ALR
w_ALR = 3*L*n_cell_H/cos(theta); %[m] width of the ALR
h_HS = 0.8*1e-3; %[m] height of the HS
w_HS = 2.5*w_ALR; %[m] width of the HS
x = [-w_HS/2;
    -w_ALR/2;
    -w_ALR/4;
    0;
    w_ALR/4;
    w_ALR/2;
    w_HS/2]'; % x-coordinates of the sampling nodes
y = [0,h_HS/2,h_HS,h_HS+h_ALR/6,h_HS+h_ALR/3,h_HS+h_ALR/2,h_HS+2*h_ALR/3,h_HS+5*h_ALR/6,h_HS+h_ALR]; % y-coordinates of the sampling nodes
[X2,Y2] = meshgrid(x,y(1:3)); % HS sampling nodes
[X1,Y1] = meshgrid(x(2:6),y(3:end)); % ALR sampling nodes
Coord = [-w_HS/2,0;
                -w_ALR/2,0;
                -w_ALR/4,0;
                0,0;
                w_ALR/4,0;
                w_ALR/2,0;
                w_HS/2,0;
                -w_HS/2,h_HS/2;
                -w_ALR/2,h_HS/2;
                -w_ALR/4,h_HS/2;
                0,h_HS/2;
                w_ALR/4,h_HS/2;
                w_ALR/2,h_HS/2;
                w_HS/2,h_HS/2;
                -w_HS/2,h_HS;
                -w_ALR/2,h_HS;
                -w_ALR/4,h_HS;
                0,h_HS;
                w_ALR/4,h_HS;
                w_ALR/2,h_HS;
                w_HS/2,h_HS;
                -w_ALR/2,h_HS+h_ALR/6;
                -w_ALR/4,h_HS+h_ALR/6;
                0,h_HS+h_ALR/6;
                w_ALR/4,h_HS+h_ALR/6;
                w_ALR/2,h_HS+h_ALR/6;
                -w_ALR/2,h_HS+h_ALR/3;
                -w_ALR/4,h_HS+h_ALR/3;
                0,h_HS+h_ALR/3;
                w_ALR/4,h_HS+h_ALR/3;
                w_ALR/2,h_HS+h_ALR/3;
                -w_ALR/2,h_HS+h_ALR/2;
                -w_ALR/4,h_HS+h_ALR/2;
                0,h_HS+h_ALR/2;
                w_ALR/4,h_HS+h_ALR/2;
                w_ALR/2,h_HS+h_ALR/2;
                -w_ALR/2,h_HS+2*h_ALR/3;
                -w_ALR/4,h_HS+2*h_ALR/3;
                0,h_HS+2*h_ALR/3;
                w_ALR/4,h_HS+2*h_ALR/3;
                w_ALR/2,h_HS+2*h_ALR/3;
                -w_ALR/2,h_HS+5*h_ALR/6;
                -w_ALR/4,h_HS+5*h_ALR/6;
                0,h_HS+5*h_ALR/6;
                w_ALR/4,h_HS+5*h_ALR/6;
                w_ALR/2,h_HS+5*h_ALR/6;
                -w_ALR/2,h_HS+h_ALR;
                -w_ALR/4,h_HS+h_ALR;
                0,h_HS+h_ALR;
                w_ALR/4,h_HS+h_ALR;
                w_ALR/2,h_HS+h_ALR]; % ordered coordinates sequence
                
% Plot ground-truth displacement field
F2 = scatteredInterpolant(Coord(1:21,1),Coord(1:21,2),comp(1:21,p)); % interpolate HS displacement field
F2.Method = 'linear';
F1 = scatteredInterpolant(Coord([16,17,18,19,20,22:end],1),Coord([16,17,18,19,20,22:end],2),comp([16,17,18,19,20,22:end],p)); % interpolate ALR displacement field
F1.Method = 'linear';
z1 = F1(X1,Y1); % ALR displacement field
figure(1);
colormap('jet');
set(gca,'DataAspectRatio',[1 1 24]);
contourf(X1,Y1,z1,20,'EdgeColor','none');
hold on;
z2 = F2(X2,Y2); % HS displacement field
contourf(X2,Y2,z2,20,'EdgeColor','none');
cb = colorbar;
title('Ground Truth [m]');
savefig('ground_truth.fig');

% Extract predicted displacement field
u_Re_WES = load("uv_val.mat","uRe"); %[m] horizontal displacement - real part
u_Im_WES = load("uv_val.mat","uIm"); %[m] horizontal displacement - imaginary part
v_Re_WES = load("uv_val.mat","vRe"); %[m] vertical displacement - real part
v_Im_WES = load("uv_val.mat","vIm"); %[m] vertical displacement - imaginary part
comp_WES = [u_Re_WES.uRe',u_Im_WES.uIm',v_Re_WES.vRe',v_Im_WES.vIm']; % list of predicted eigenvector components

% Plot predicted displacement field
Q2 = scatteredInterpolant(Coord(1:21,1),Coord(1:21,2),comp_WES(N*K+(1-1)*400:400:N*K+(21-1)*400,p)); % interpolate HS displacement field
Q2.Method = 'linear';
Q1 = scatteredInterpolant(Coord([16,17,18,19,20,22:end],1),Coord([16,17,18,19,20,22:end],2),comp_WES([N*K+(16-1)*400,N*K+(17-1)*400,N*K+(18-1)*400,N*K+(19-1)*400,N*K+(20-1)*400,N*K+(22-1)*400:400:N*K+(51-1)*400],p)); % interpolate ALR displacement field
Q1.Method = 'linear';
z1WES = Q1(X1,Y1); % ALR displacement field
figure(2);
colormap('jet');
set(gca,'DataAspectRatio',[1 1 24]);
contourf(X1,Y1,z1WES,20,'EdgeColor','none');
hold on;
z2WES = Q2(X2,Y2); % HS displacement field
contourf(X2,Y2,z2WES,20,'EdgeColor','none');
cb = colorbar;
title('Predicted [m]');
savefig('predicted.fig');
