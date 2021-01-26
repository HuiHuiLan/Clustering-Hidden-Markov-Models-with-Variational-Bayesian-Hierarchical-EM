% demo_faces - example of eye gaze analysis for face recognition 
%

clear
close all

%% Load data from xls %%%%%%%%%%%%%%%%%%%%%%%%%%
% see the xls file for the format
[data, SubjNames, TrialNames] = read_xls_fixations('demodata.xls');
%[data, SubjNames, TrialNames] = read_xls_fixations('jov.xlsx');

%% VB Parameters for HMMs %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% faceimg = 'face.jpg';
faceimg = 'ave_face120.png';

img0 = imread(faceimg);
imgsize = size(img0);


S = 1:3; % automatically select from K=1 to 3
vbopt.alpha0 = 1;
vbopt.mu0    = [imgsize(2); imgsize(1)]/2; 
vbopt.W0     = 0.001;
vbopt.beta0  = 1;
vbopt.v0     = 10;
vbopt.epsilon0 = 1;
vbopt.showplot = 0;     % set this to 1 to see each subject during learning
vbopt.bgimage = faceimg;

vbopt.seed = 100;  % set random state seed for reproducible results.

% Estimate hyperparameters automatically for each individual
% (remove this option to use the above hyps)
vbopt.learn_hyps = 1;  

%% Learn Subjects' HMMs %%%%%%%%%%%%%%%%%%%%%
% estimate HMMs for each individual

[hmms, Ls] = vbhmm_learn_batch(data, S, vbopt);

%% Cluster HMMs 

fprintf('=== Clustering ===\n');

K = 1:5;

% same as vbhmm_learn
vbhemopt.alpha0 = 1;
vbhemopt.eta0 = 1;
vbhemopt.m0    = vbopt.mu0; 
vbhemopt.W0     = 0.001;
vbhemopt.lambda0  = 1;   % is beta0 in vbhmm_learn
vbhemopt.v0     = 10;
vbhemopt.epsilon0 = 1;

vbhemopt.initmode= 'wtkmeans';
vbhemopt.seed = 1001; 
vbhemopt.Nv =10; 
vbhemopt.tau=5;
vbhemopt.trials = 50;

group_vbhmms = vbhem_h3m_cluster(hmms, K, S, vbhemopt);  % K groups, S hidden states


% prunning out hmms/states with small prior
[group_vbhmms_clear] = vbh3m_remove_empty(group_vbhmms);
% vhem_plot need hmms, not hmm
group_vbhmms_clear.hmms = group_vbhmms_clear.hmm;

vhem_plot(group_vbhmms_clear, faceimg);

plot(group_vbhmms_clear.model_LL,'LineWidth',1.5)

title(['Model Selection '])
xlabel('K');
ylabel('ELBO');
set(gca,'linewidth',1,'fontsize',20,'fontname','Times');



