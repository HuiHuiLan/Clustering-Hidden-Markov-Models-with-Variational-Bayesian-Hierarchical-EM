function [h3m_r]= vbhem_h3m_cluster(hmms, K, S, vbhemopt)
% main function 
% Using VBHEM algorithm cluster HMMs
%
% INPUTS
%      hmms = 1xN cell array of HMMs, learned with vbhmm_learn.
%
%         K = number or vecter of clusters (required)  
%         S = number or vecter of states (ROIs) in one HMM     
%         vector: automatically selects the number of hidden states from the given values. 
%                 The model (K,S) with highest log-likelihood will be selected.
%             
%
%  vbhembopt = structure containing other options as below: (similar with vbhmm_learn)
%  
%
%   VB hyper-parameters: each hmm in h3m_b has the same initial hyper-parameters
%     vbhemopt.alpha = Dirichlet distribution (for mixing coefficient of h3m) concentration parameter -- large value
%                   encourages uniform prior, small value encourages concentrated prior (default=0.1)
%                   Another way to think of it is in terms of "virtual" samples -- A typical way to 
%                   estimate the probability of something is to count the number of samples that it
%                   occurs and then divide by the total number of samples, 
%                   i.e. # times it occurred / # samples.  
%                   The alpha parameter of the Dirichlet adds a virtual sample to this estimate, 
%                   so that the probability estimate is (# times it occurred + alpha) / # samples.  
%                   So for small alpha, then the model will basically just do what the data says.  
%                   For large alpha, it forces all the probabilities to be very similar (i.e., uniform).
%     vbhemopt.eta  = Dirichlet distribution (for initial probabilities)
%     concentration parameter, similar to alpha.
%     vbhemopt.epsilon = Dirichlet distribution for rows of the transition matrix (default=1). 
%                     The meaning is similar to alpha, but for the transition probabilities.
% 
%     vbhemopt.mu   = prior mean - it should be dimension D;
%                   for eye gaze project  
%                   for D=2 (fixation location), default=[256;192] -- typically it should be at the center of the image.
%                   for D=3 (location & duration), default=[256,192,250]
%     vbhemopt.W    = size of the inverse Wishart distribution (default=0.005).
%                   If a scalar, W is assumed to be isotropic: W = vbhemopt.W*I.
%                   If a vector, then W is a diagonal matrix:  W = diag(vbhemopt.W);
%                   This determines the covariance of the ROI prior.
%                   For example W=0.005 --> covariance of ROI is 200 --> ROI has standard devation of 14.
%                   
%     vbhemopt.v     = dof of inverse Wishart, v > D-1. (default=5) -- larger values give preference to diagonal covariance matrices.
%     vbhemopt.lambda  = Wishart concentration parameter -- large value encourages use of
%                   prior mean & W, while small value ignores them. (default=1)
 
%   VBHEM Algorithm parameters
%     vbhemopt.initmode     = initialization method (default='baseem')
%                             follow the initialization ways in VHEM, using
%                             the parameters in the h3m to update the
%                             hyperparameters.
%                             'wtkmeans' - using weight k-means to cluster
%                             data as initialization
%                            'random' - randomly set assignment varibles z and phi 
%                            'baseem' - randomly select a set of base
%                              emissions
%                            'inith3m' - specify a H3M as the initialization
%                            'gmmNew', 'gmmNew2' - initialize emissions
%                            using HEM-G3M estimated
%     vbhemopt.trials    = number of trials for 'random' initialization (default=50)
                              
%     vbhemopt.maxIter      = max number of iterations (default=100)
%     vbhemopt.minDiff      = tolerence for convergence (default=1e-5)
%     vbhemopt.showplot     = show plots (default=0)
%     vbhemopt.sortclusters = '' - no sorting [default]
%                          'e' - sort ROIs by emission probabilites
%                          'p' - sort ROIs by prior probability
%                          'f' - sort ROIs by most-likely fixation path [default]
%                              (see vbhmm_standardize for more options)
%     vbhemopt.verbose   = 0 - no messages
%                        = 1 - a few messages showing progress [default]
%                        = 2 - more messages
%                        = 3 - debugging
%
% OUTPUT  (need modify)
%   h3m_r = structure containing the hmms in the reduced model and related information
%   h3m_r.K                  = the selected number of clusters/components
%   h3m_r.S                  = the selected number of states
%   h3m_r.omega              = omega [1 x Kr]
%   h3m_r.hmm                = the hmms in h3m_r, cell [1 x Kr]
%   h3m_r.hmm{j}.varpar      = the updated variational parameters
%   h3m_r.hmm{j}.prior       = prior probababilies [Sr x 1]
%   h3m_r.hmm{j}.trans       = transition matrix [Sr x Sr]: trans(i,j) is the P(x_t=j | x_{t-1} = i)
%   h3m_r.hmm{j}.pdf{j}.mean = ROI mean [1 x D]
%   h3m_r.hmm{j}.pdf{j}.cov  = covariances [D x D]
%   h3m_r.hmm{j}.M           = transition counts [K x K]
%   h3m_r.hmm{j}.N1          = prior counts [K x 1]
%   h3m_r.hmm{j}.N           = cluster sizes [K x 1] (emission counts)
%   h3m_r.Z                  = the assignment variable
%   h3m_r.alpha              = the updated alpha
%   h3m_r.Nj                 = the number of base hmms assigned to j-th cluster
%   h3m_r.W0mode             = the mode of W0 -iid: one scaler
%   h3m_r.L_elbo             = be used to compute DIC
%   h3m_r.LL                 = ELBO
%   h3m_r.learn_hyps         = the structure contains option and results of learned hyperpars
%   h3m_r.label              = the cluster assignment for each input HMM [1xN]
%   h3m_r.groups             = indices of input HMMs in each group {1xK}
%   h3m_r.group_size         = size of each group [1xK]
%   group_hmm.LogLs          = log-likelihood score
%
%% set default values

if nargin<4
  vbhemopt = struct;
end

% number of clusters
vbhemopt.K = K;

% number of states 
vbhemopt.S = S;

vbhemopt = setdefault(vbhemopt, 'verbose', 1);
vbhemopt = setdefault(vbhemopt, 'remove_empty', 1);

if (vbhemopt.remove_empty)
  if (vbhemopt.verbose)
    fprintf('Checking input HMMS: ');
  end
  for i=1:length(hmms)
    [hmms{i}, zi] = vbhmm_remove_empty(hmms{i}, 0, 1e-3);
    if ~isempty(zi)
      if (vbhemopt.verbose)
        fprintf('%d: removed states', i);
        fprintf(' %d', zi);
        fprintf('; ');
      end
    end
    
  end
  if (vbhemopt.verbose)
    fprintf('done\n');
  end
end


% assume the data has same dimensions
D = size(hmms{1}.pdf{1}.mean,2);
switch(D)    
  case 2
    defmu = [256;192]; % for face data
    %defmu = zeros(D,1);
  case 3
    defmu = [256;192;150];
  otherwise
    defmu = zeros(D,1);
    %warning(sprintf('no default mu0 for D=%d', D));
end

vbhemopt = setdefault(vbhemopt, 'alpha0', 1); 
vbhemopt = setdefault(vbhemopt, 'eta0', 1); 
vbhemopt = setdefault(vbhemopt, 'epsilon0', 1);
vbhemopt = setdefault(vbhemopt, 'm0',   defmu); % hyper-parameter for the mean
vbhemopt = setdefault(vbhemopt, 'W0',     0.005); % the inverse of the variance of the dimensions
vbhemopt = setdefault(vbhemopt, 'lambda0',  1);
vbhemopt = setdefault(vbhemopt, 'v0',     5);

% setable parameters (could be changed)
vbhemopt = setdefault(vbhemopt, 'trials', 100);     % number of trials to run  in vhem is 'numtrials'
vbhemopt = setdefault(vbhemopt, 'termmode', 'L');   % rule to terminate EM iterations
vbhemopt = setdefault(vbhemopt, 'termvalue', 1e-5);
vbhemopt = setdefault(vbhemopt, 'max_iter', 200);    % max number of iterations
vbhemopt = setdefault(vbhemopt, 'min_iter', 1);     % min number of iterations
vbhemopt = setdefault(vbhemopt, 'sortclusters', 'f');
vbhemopt = setdefault(vbhemopt, 'initmode',  'auto');  % initialization mode
vbhemopt = setdefault(vbhemopt, 'minDiff',   1e-5);
vbhemopt = setdefault(vbhemopt, 'showplot',  0);
vbhemopt = setdefault(vbhemopt, 'random_gmm_opt', {});


% standard parameters (not usually changed)
vbhemopt = setdefault(vbhemopt, 'Nv', 100); % number of virtual samples
vbhemopt = setdefault(vbhemopt, 'tau', 10); % temporal length of virtual samples
vbhemopt = setdefault(vbhemopt, 'initopt', struct);   % options for 'gmm' initaliation (unused)
vbhemopt.initopt = setdefault(vbhemopt.initopt, 'iter', 30);  % number of GMM iterations for init (unused)
vbhemopt.initopt = setdefault(vbhemopt.initopt, 'trials', 4); % number of GMM trials for init (unused)
vbhemopt.initopt = setdefault(vbhemopt.initopt, 'mode', ''); % other options

vbhemopt = setdefault(vbhemopt, 'learn_hyps', 1);

% minimum/maximum values of hyperparameters, for numerical stability
vbhemopt = setdefault(vbhemopt, 'hyps_max', struct);
hyps_max = vbhemopt.hyps_max;
hyps_max = setdefault(hyps_max, 'alpha0',   1.0686e+13);  % exp(30)
hyps_max = setdefault(hyps_max, 'eta0',   1.0686e+13);  % exp(30)
hyps_max = setdefault(hyps_max, 'epsilon0', 1.0686e+13);  % exp(30)
hyps_max = setdefault(hyps_max, 'v0',       1e4);
hyps_max = setdefault(hyps_max, 'lambda0',    1.0686e+13);  % exp(30)
hyps_max = setdefault(hyps_max, 'W0',       1.0686e+13);  % exp(30)
vbhemopt.hyps_max = hyps_max;

vbhemopt = setdefault(vbhemopt, 'hyps_min', struct);
hyps_min = vbhemopt.hyps_min;
hyps_min = setdefault(hyps_min, 'alpha0',    1.0686e-13);  % exp(-30)
hyps_min = setdefault(hyps_min, 'eta0',    1.0686e-13);  % exp(-30)
hyps_min = setdefault(hyps_min, 'epsilon0', 1.0686e-13);  % exp(-30)
hyps_min = setdefault(hyps_min, 'v0',       2.0612e-09+D-1);  % exp(-20)+dim-1 - should be okay to ~10000 dimensions
hyps_min = setdefault(hyps_min, 'lambda0',    1.0686e-13);  % exp(-30)
hyps_min = setdefault(hyps_min, 'W0',       1.0686e-13);  % exp(-30)
vbhemopt.hyps_min = hyps_min;


vbhemopt = setdefault(vbhemopt, 'calc_LLderiv', 0);
vbhemopt = setdefault(vbhemopt, 'keep_suboptimal_hmms', 0);
vbhemopt = setdefault(vbhemopt, 'minimizer', 'minimize-lbfgs');
vbhemopt = setdefault(vbhemopt, 'canuseMEX', 1);
vbhemopt = setdefault(vbhemopt, 'verbose_prefix', '');
vbhemopt = setdefault(vbhemopt, 'keep_best_random_trial', 1);
vbhemopt = setdefault(vbhemopt, 'seed', []);
vbhemopt = setdefault(vbhemopt, 'use_post', 1);


if isempty(vbhemopt.initopt.mode)
  % set defaults for each initialization type
  switch(vbhemopt.initmode)
    case 'baseem'
      vbhemopt.initopt.mode = 'u';
    case {'gmmNew','wtkmeans','con-wtkmeans'}
      vbhemopt.initopt.mode = 'r0';
    case {'gmmNew2','gmmNew2_r'}
      vbhemopt.initopt.mode = 'u0';  
  end
end

  
% fixed parameters (not changed)
vbhemopt.emit.type= 'gmm';
vbhemopt.emit = setdefault(vbhemopt.emit, 'covar_type', 'full');
vbhemopt.M = 1; % number of mixture components in each GMM for an emission (should always be 1)

% check validity
if vbhemopt.v0 <= D-1
  error('v0 not large enough...should be > D-1');
end

% convert list of HMMs into an H3M
h3m_b = hmms_to_h3m_hem(hmms, vbhemopt.emit.covar_type, vbhemopt.use_post);

% check seed is set
if isempty(vbhemopt.seed)
  error('vbhemopt.seed has not been set. This is required to obtain reproducible results.');
end

% check for estimating hyps automatically
if iscell(vbhemopt.learn_hyps) || (vbhemopt.learn_hyps == 1)
  do_learn_hyps = 1;
else
  do_learn_hyps = 0;
end

VERBOSE_MODE = vbhemopt.verbose;
  
%% run for multiple K, single S %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%- consider different conditions for K & S
%-- if multiple K
%----if multiple S
%-- else sigle K & sigle S
% Hence the base condition is sigle K & single S, others just call this base
% condition multiple times 

if length(K)>1
    % call learning for each value of K
    out_all = cell(1,length(K));
    LLk_all  = zeros(1,length(K));
    for ki = 1:length(K)
        vbhemopt1 = vbhemopt;
        vbhemopt1.showplot = 0; 
        if (VERBOSE_MODE >= 2)
            fprintf('-- K=%d --\n', K(ki));
        elseif (VERBOSE_MODE == 1)
            fprintf('-- VBH3M K=%d: ', K(ki));
        end

        % call learning with a single k
        out_all{ki} = vbhem_h3m_cluster(hmms, K(ki), S, vbhemopt1);
        LLk_all(ki) = out_all{ki}.LL;
    end
        
    % correct for multiple parameterizations
    LLk_all = LLk_all + gammaln(K+1);
    
    % get K with max data likelihood
    [maxLLk,indk] = max(LLk_all);

    % return the best model
    h3m             = out_all{indk};
    h3m.model_LL    = LLk_all;
    h3m.model_k     = K;
    h3m.model_bestK = K(indk);
    h3m.model_all   = out_all;
    %L = maxLLk;

    if VERBOSE_MODE >= 1
        if length(S)>1
            fprintf('best model: K=%d; S=%d; L=%g\n', K(indk), h3m.model_bestS, maxLLk);
        else
            fprintf('best model: K=%d; L=%g\n', K(indk), maxLLk);
        end
        if strcmp(vbhemopt.initmode, 'auto')
            fprintf('best init was %s',  h3m.Initmodes);
        end
        if (do_learn_hyps)
            fprintf('  ');
            vbhmm_print_hyps({h3m.learn_hyps.hypinfo.optname}, h3m.learn_hyps.vbopt);
            fprintf('\n');
        end    
    end

    h3m_out = h3m;
    
else   
    %% run for a single K %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if length(S)>1       
        %% run for multiple S
             
        % call learning for each value of S
        out_all_s = cell(1,length(S));
        LLs_all   = zeros(1,length(S));
        %for si = 1:length(S)
        for si = 1:length(S)
              %vbhemopt2 = vbhemopt;
            if (VERBOSE_MODE >= 2)
                fprintf('-- S=%d --\n', S(si));
            elseif (VERBOSE_MODE == 1)
                fprintf('-- VBH3M K=%d--S=%d: ', K,S(si));
            end
    
            % call learning with a single k
            out_all_s{si} = vbhem_h3m_cluster(hmms, K, S(si), vbhemopt);
            LLs_all(si)   = out_all_s{si}.LL;
        end

        % correct for multiple parameterizations
        LLs_all = LLs_all + gammaln(S+1);
  
        % get K with max data likelihood
        [maxLLs,inds] = max(LLs_all);

        % return the best model
        h3m             = out_all_s{inds};
        h3m.model_LL_S  = LLs_all;
        h3m.model_S     = S;
        h3m.model_bestS = S(inds);
        h3m.model_all_s = out_all_s;
        %L = maxLLs;

        if VERBOSE_MODE >= 1
          fprintf('best model: S=%d; L=%g\n', S(inds), maxLLs);
          if strcmp(vbhemopt.initmode, 'auto')
            fprintf('best init was %s\n',  h3m.Initmodes);
          end
  
        end
        h3m_out = h3m;

    else
   %% run for a single S and single K

      if ~strcmp(vbhemopt.initmode, 'auto')          
        % run VBHEM clustering        
         [h3m_out] = vbhem_h3m_c(h3m_b, vbhemopt);
         
      else 
  %% auto initialization - try these methods

         if isfield(vbhemopt,'initmodes')
             initmodes = vbhemopt.initmodes; 
             initopts = vbhemopt.initopts;
         else
             %
             initmodes = { 'baseem', 'gmmNew','wtkmeans'};
             initopts = { 'u', 'r0', 'r0'};
         end


         for ii=1:length(initmodes)   
             tic;
             vbhemopt3 = vbhemopt;
             vbhemopt3.initmode = initmodes{ii};
             vbhemopt3.initopt.mode = initopts{ii};
             fprintf('auto initialization: trying %s: ', vbhemopt3.initmode);
             h3m_out_trials{ii} = vbhem_h3m_c(h3m_b, vbhemopt3);
             h3m_out_trials{ii}.Initmodes = initmodes{ii};
             trials_LL(ii) = h3m_out_trials{ii}.LL;            
             h3m_out_trials{ii}.time = toc;
         end
  
         % get best method
         [maxLL, ind] = max(trials_LL);
         h3m_out = h3m_out_trials{ind};
         if vbhemopt.keep_best_random_trial    
            h3m_out.h3m_out_trials = h3m_out_trials;
         end

         fprintf('best init was %s; LL=%g\n', initmodes{ind}, trials_LL(ind));
         
         % correct for multiple parameterization
         h3m_out.LL_orignal = h3m_out.LL;
      end
    
    end %END FOR S
end  %END FOR K

% if ~isempty(vbhemopt.sortclusters)
%     for j = 1:h3m_out.K 
%         tmphmm = h3m_out.hmm{j};
%         tmphmm = vbhmm_standardize(tmphmm, vbhemopt.sortclusters);
%         h3m_out.hmm{j} = tmphmm;
%     end
% end

h3m_r = h3m_out;
% if strcmp(vbhemopt.initmode, 'auto') 
% if vbhemopt.keep_best_random_trial
%     
%     h3m_r.h3m_out_trials = h3m_out_trials;
% end
% end
%[h3m_r] = vbh3m_remove_empty(h3m_out);

% append the options
h3m_r.vbhemopt = vbhemopt;
  

function hemopt = setdefault(hemopt, field, value)
if ~isfield(hemopt, field)
  hemopt.(field) = value;
end

