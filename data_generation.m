close all
clear all
clc

%==========================================%
%                                          %
%   DeepF-fNet: tri-chiral honeycomb LRM   %
%                                          %
%       Data generation with Comsol        %
%                                          %
%         Author: Andrea Tollardo          %
%                                          %
%==========================================%

fprintf('Loading Comsol model ...\n');
model = mphload('Direct_Band-gap_Problem2D'); % Load Comsol model

% PARAMETERS RANGES AND DATASET POPULATION TO CHANGE
%===========================================
r = linspace(0.04,4,100)*1e-3; % [m] - Geometrical parameter r
s = linspace(0.04,4,100)*1e-3; %[m] - Geometrical parameter s
L = linspace(0.04,4,100)*1e-3; %[m] - Geometrical parameter L
N_max = 2000 % dataset population
%===========================================
i_r = randperm(length(r)); % shuffle r randomly
r(i_r) = r; % shuffle r randomly
i_s = randperm(length(s)); % shuffle s randomly
s(i_s) = s; % shuffle s randomly
i_L = randperm(length(L)); % shuffle L randomly
L(i_L) = L; % shuffle L randomly

k = linspace(-1,1,40); %[-] - Sweep Study parameter
data_size = int32(length(r)*length(L)*length(th)); % Max number of direct-problem simulations that could be run
TrainingTensor_nr = zeros([51,848,1]); % initialize training tensor
time0 = datetime; % start computation time monitoring
id = 0; % initialize iteraration index
id_possible = 0; % initialize remaining iterations
fprintf('\nStarting Parametric Sweep ...\n');
while id < N_max
	for ii=1:length(r)
	    for jj=1:length(L)
		for kk=1:length(s)
		    id_possible = id_possible +1; % increase number of remaining iterations
		    if r(ii) <= s(kk)/2 || L(jj)/2<=r(ii)+s(kk)/2
		        continue % skip combination if compliance constraints are not respected
		    end
		    id = id + 1; % increase number of good combinations
		    
    		    % Set parameters for Comsol study
		    r_new = sprintf('%f [m]', r(ii));
		    L_new = sprintf('%f [m]', L(jj));
		    s_new = sprintf('%f [m]', s(kk));
		    model.param.set('r',r_new);
		    model.param.set('L',L_new);
		    model.param.set('s',s_new);
		    
		    % Generate set of coordinates for these geometric parameters
		    theta = atan(2*r(ii)/L(jj)); %[rad] ligament angle of tri-chiral honeycomb
		    n_cell_H = 5; % number of horizontal repetitions of hexagons
		    n_cell_V = 9; % number of vertical repetitions of hexagons
		    h_ALR = sqrt(3)*L(jj)*n_cell_V/cos(theta); %[m] height of the ALR 
		    w_ALR = 3*L(jj)*n_cell_H/cos(theta); %[m] width of the ALR 
		    h_HS = 0.8*1e-3; %[m] height of the HS 
		    w_HS = 2.5*w_ALR; %[m] width of the HS
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
				w_ALR/2,h_HS+h_ALR]'; % ordered coordinates sequence
		    
		    % Parametric Sweep Study
		    fq = zeros([10,40]); % initialize the dispersion curve
		    u = zeros([51,400]); % initialize horizontal nodal displacement matrix
		    v = zeros([51,400]); % initialize vertical nodal displacement matrix
		    uv_FEM = zeros([51,800]); % initialize nodal displacement matrix
		    for uu=1:length(k) % sweep the discretized wave number range
		        k_new = sprintf('%f',k(uu)); % set wave number
		        model.param.set('k',k_new);
		        model.study('std1').run; % run Comsol study
		        
		        % Save dispersion curve
		        fq(:,uu) = real(mphglobal(model,'freq')); %[Hz] dispersion curve
		        
		        % Save displacements
		        u(:,((uu-1)*10+1):uu*10) = (mphinterp(model,{'comp1.u'},"coord",Coord))'; %[m] horizontal nodal displacements
		        v(:,((uu-1)*10+1):uu*10) = (mphinterp(model,{'comp1.v'},"coord",Coord))'; %[m] vertical nodal displacements
		    end
		    
		    % Populate nodal displacement matrix
		    uv_FEM(:,1:2:end) = u; % horizontal displacement
		    uv_FEM(:,2:2:end) = v; % vertical displacement
		    omega2 = (2*pi*fq).^2; %[rad2/s2] convert dispersion curve to SI units
		    kx = k*pi/w_HS; %[rad/m] discretized wave number range

		    % Build the Training Tensor
		    TrainingTensor_nr(3,1,id) = w_HS; % save HS width, a.k.a. minimum wave length
		    TrainingTensor_nr(2,1:3,id) = [r(ii),L(jj),s(kk)]; % save IEPS labels
		    TrainingTensor_nr(1:10,4:43,id) = omega2; % save IEPS input
		    TrainingTensor_nr(1:40,44,id) = kx; % save ground truth wave number range
		    TrainingTensor_nr(:,47:48,id) = Coord'; % save ground truth coordinates
		    TrainingTensor_nr(:,49:end,id) = uv_FEM; % save WES labels
		    TrainingTensor_nr = cat(3,TrainingTensor_nr,zeros(51,848,1)); % prepare dataset to next iteration

		    fprintf(['___________________________________________________\n' ...
		        'Iteration %d Summary:\n' ...
		        '- computation time: %s\n' ...
		        '- max remaining iterations: %d\n' ...
		        '...\n'],id,datetime-time0,data_size-id_possible);
		end
	    end
	end
end

% Shuffle batches randomly
fprintf('\nShuffling bacthes randomly ...\n');
idx = randperm(id); % random permutation
TrainingTensor(:,:,idx) = TrainingTensor_nr(:,:,1:end-1); % exclude last empty dataset element

% Save TrainingTensor for NN training
fprintf('\nSaving training tensor to TrainingTensorNN.mat ...\n');
save('TrainingTensorNN','TrainingTensor','-v7.3');
fprintf('\nData generation completed\n');
