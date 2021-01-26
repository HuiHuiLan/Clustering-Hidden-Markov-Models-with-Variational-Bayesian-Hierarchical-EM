%% The journal synthetic experiments
%% experiment 1
%% generate synthetic data

outdir = 'exprmt1_results/';
%outdir_fix              = 'VBHMMs/';
[success, msg] = mkdir(outdir);

start_iter = 1;
iter = 10;

num_hmm = 2;  %the number of the initial clusters hmm
num_sta = 2;  % the number of each hmm
N = 20;   % the number of hmms for each cluster
dim =2;
Kb = num_hmm*N;
T= 50;   % the length of sequences
Nn = 25;

%% set the GT hmms
pr1 = [1/2,1/2];
tr1 =[0.6,0.4;0.4,0.6];
m1 =  [0,0];
m2 =  [3,3];
cov1 = [1,1];
cov2 = [1,1];

hmm1.prior =pr1;
hmm1.trans = tr1;
hmm1.pdf{1}.mean = m1;
hmm1.pdf{2}.mean = m2;
hmm1.pdf{1}.cov = diag(cov1);
hmm1.pdf{2}.cov = diag(cov2);

pr2 =[1/2,1/2];
tr2 =[0.4,0.6;0.6,0.4];

hmm2.prior =pr2;
hmm2.trans = tr2;
hmm2.pdf{1}.mean = m1;
hmm2.pdf{2}.mean = m2;
hmm2.pdf{1}.cov = diag(cov1);
hmm2.pdf{2}.cov = diag(cov2);


%% LEARN & CLUSTER

%starttime = clock;
Adata = [];

for it =start_iter:iter
   
rng(it, 'twister');
%% sample data
data11 = {};
data22 = {};

for i =1:N
    data1={};
    for j = 1:Nn
        [~, data1{j}] = vbhmm_random_sample(hmm1, T, 1);
        a =  cell2mat(data1{j}) +normrnd(0,0.1,[T,1]);
        data1(j) = mat2cell(a,T);       
    end
    data11{i} =  data1';
end


data1 = data11';

for i =1:N
    data2={};
    for j = 1:Nn
        [~, data2{j}] = vbhmm_random_sample(hmm2, T, 1);
        a =  cell2mat(data2{j}) +normrnd(0,0.1,[T,1]);
        data2(j) = mat2cell(a,T);       
    end
    data22{i} =  data2';
end

data2 = data22';


data_test = [data1;data2];

Adata{it} = data_test;
end

defmu = [1.5;1.5];
defW = 1;

outfile = [outdir 'data.mat'];
fprintf('saving data MAT files: %s\n', outfile);
save(outfile,'Adata','defmu', 'defW','num_hmm','num_sta','N','dim','Kb','T');





