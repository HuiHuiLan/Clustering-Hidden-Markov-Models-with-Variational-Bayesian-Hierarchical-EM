%% The journal synthetic experiments
%% experiment 1

% clear 
% close all
run exprmt1_sampledata.m

% the loop times of experiments
start_iter = 1;
iter = 1;


% load data
% load('Adata.mat')
%% setup directories
outdir = 'exprmt1_results/';
%outdir_fix              = 'VBHMMs/';
[success, msg] = mkdir(outdir);
%[success, msg] = mkdir(outdir, outdir_fix);

%%

N = 20;   % the number of hmms for each cluster
dim =2;
Kb = num_hmm*N;
Nv =100;

%% LEARN & CLUSTER

%% LEARN HMM

vbopt =struct;
vbopt.mu0 = defmu; % hyper-parameter for the mean
vbopt.W0 = defW;
vbopt.showplot = 0;     % set this to 1 to see each subject during learning

vbopt.seed = 1001;  % set random state seed for reproducible results.
vbopt.learn_hyps = 1;  

S = 2;

starttime = clock;

for it =start_iter:iter
  
    data_test = Adata{it};
    [hmms, Ls] = vbhmm_learn_batch(data_test, S, vbopt);

    HMMs{it} = hmms;
end

stoptime = clock;

VBHMMspend_time = etime(stoptime,starttime);


%% save mat file
outfile = [outdir 'HMMs.mat'];
fprintf('saving individual MAT files: %s\n', outfile);
save(outfile,'HMMs','Kb'); %,'VBHMMspend_time'

%% %% CLUSTER HMMs VBHEM

K = 1:6; % the number of hmms
S = 1:5; % the number of stats

Nv =100;

vbhemopt =struct;
vbhemopt.Nv =Nv; % number of virtual samples
vbhemopt.tau = T; % temporal length of virtual samples
vbhemopt.alpha0 = 1e6;
vbhemopt.eta0 = 1;
vbhemopt.epsilon0 = 1;
vbhemopt.lambda0 = 1;
vbhemopt.v0 = 5;
vbhemopt.W0 = defW; % the inverse of the variance of the dimensions
vbhemopt.m0 = defmu;
vbhemopt.initmode= 'baseem';

starttime = clock;

for it =start_iter:iter

fprintf('=== Clustering ===\n');
fprintf('=== LOOP %d ===\n', it);

hmms = HMMs{it};
  
vbhemopt.seed = 1001+50*(it-1);

group_vbhmms{it} = vbhem_h3m_cluster(hmms, K, S, vbhemopt);

%% save results

matfile_cluster = [outdir,'syn_exp1_vb_' mat2str(it) '.mat'];

matfile_individual = ['vbhmms' mat2str(it)];
% 
eval([matfile_individual ,'=','group_vbhmms{it}',';']);

save(matfile_cluster, matfile_individual);

end

stoptime = clock;

VBHEMspend_time = etime(stoptime,starttime);
% 
outfile = [outdir 'VBgroup_hmms.mat'];
fprintf('saving individual MAT files: %s\n', outfile);
save(outfile,'group_vbhmms','K','S');

%% CLUSTER with VHEM
hemopt = struct;
hemopt.tau = T; 
hemopt.Nv = Nv; % number of virtual samples
hemopt.initmode= 'auto';

starttime = clock;

for it =start_iter:iter
     % set the virtual sequence length to the median data length.
    hemopt.seed = 1001+it;  % set random state seed for reproducible results.
    hmms = HMMs{it};
   % data = Adata{it};
    
    for k =1:length(K)
        for s = 1:length(S)
            all_hmms{k,s} = vhem_cluster(hmms, K(k), S(s), hemopt);  % 1 group, 3 hidden states
            
        end
    end
    vh_hmms{it} = all_hmms;
    
    matfile_cluster = [outdir,'syn_exp1_vh_' mat2str(it) '.mat'];
    
    matfile_individual = ['vhhmms' mat2str(it)];
    %
    eval([matfile_individual ,'=','vh_hmms{it}',';']);
    
    save(matfile_cluster, matfile_individual);
    
end

stoptime = clock;

VHEMspend_time = etime(stoptime,starttime);

outfile = [outdir 'VHHMMs.mat'];
fprintf('saving individual MAT files: %s\n', outfile);
save(outfile,'vh_hmms');


%% CCFD CLUSTER

slope = 3;

for it =start_iter:iter
    
    hmms = HMMs{it};
    data = Adata{it};
    try
        [CCFD_HMMs] = myccfd(hmms,data,slope);
    catch ME
        CCFD_HMMs=[];
    end
    AllCCFD_HMMs{it} = CCFD_HMMs;
    
end

% plot one results
CCFD_plot(CCFD_HMMs.ccfd_result.rho,CCFD_HMMs.ccfd_result.delta,slope,CCFD_HMMs.ccfd_result.NCLUST,CCFD_HMMs.ccfd_result.icl)


outfile = [outdir 'CCFD.mat'];
fprintf('saving individual MAT files: %s\n', outfile);
save(outfile,'AllCCFD_HMMs');

%% PPK % need long time
%% learn HMMs for each S
vbopt =struct;
vbopt.mu0 = defmu; % hyper-parameter for the mean
vbopt.W0 = defW;
vbopt.showplot = 0;     % set this to 1 to see each subject during learning

vbopt.seed = 1001;  % set random state seed for reproducible results.
vbopt.learn_hyps = 1;
PPK_indhmms = {};
for it = start_iter:iter
        
    data_test = Adata{it};
    starttime = clock;

    for s = 1:length(S)
        fprintf('=== LOOP %d, %d===\n', it,s);            
        [hmms] = vbhmm_learn_batch(data_test, S(s), vbopt);        
        PPK_indhmms{it,s} = hmms;        
    end    
    stoptime = clock;    
    PPKspend_time = etime(stoptime,starttime);
    
end


for it = start_iter:iter
    
    fprintf('=== Clustering ===\n');
    fprintf('=== LOOP %d, %d ===\n',  it);
    
    PPK_h3m = [];
    for kk= 1:length(K)        
        for ss = 1:length(S)           
            hmms = PPK_indhmms{it,ss};
            PPK_h3m{kk,ss} =ppk_sc(hmms,K(kk));
        end        
    end
    
    APPK_h3m{it} = PPK_h3m;    
end



stoptime = clock;
PPKspend_time = etime(stoptime,starttime);

PPK_LL = zeros(length(K),length(S),iter);
for it = start_iter:iter
    %fprintf('=== Clustering ===\n');
    fprintf('=== LOOP %d, %d ===\n',  it);

    data_test = Adata{it};
    
    PPK_h3m = APPK_h3m{it};
    
    for kk = 1:length(K)
        Kgroup = K(kk);
        
        for ss = 1:length(S)

            tmph3m = PPK_h3m{kk,ss};
            
            Group_data = cat(1,data_test{:});
            loglik_hmm = zeros(length(Group_data),Kgroup);
            for k = 1:Kgroup
                [tmphmms] = vbhmm_remove_empty(tmph3m.hmms{k}, 0, 1e-3);

                [loglik_hmm(:,k)]= vbhmm_ll(tmphmms,Group_data);
            end
                        
            PPK_LL(kk,ss,it)  = sum(logtrick(log(tmph3m.weight) + loglik_hmm'));
        end
    end    
end

outfile = [outdir 'PPK.mat'];
fprintf('saving individual MAT files: %s\n', outfile);
save(outfile,'APPK_h3m','PPK_LL');


