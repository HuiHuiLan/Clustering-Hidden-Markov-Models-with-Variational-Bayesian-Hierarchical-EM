function [ P_d,DIC] = myDIC(hmms, h3m,T,dt,issyn)
% this function to compute the DIC
% hmms are the h3m_b
% h3m is h3m_r

if nargin <4
    dt=0;
end
if nargin <5
    issyn = 0;
end

if isfield(h3m,'NL')
    N_supj = h3m.NL;
else
    N_supj = h3m.Nj;
end

K = h3m.K;
Kb = length(hmms);
Ni = sum(N_supj)/Kb;
H3M = hmms_to_h3m(hmms, 'diag');


gamma0 = h3m.learn_hyps.vbopt.lambda0;

dim = length(h3m.hmm{1, 1}.pdf{1, 1}.mean);

% omega
if isfield(h3m,'alpha')
    alpha = h3m.alpha;
else
    
    alpha = h3m.varpar.alpha;
end

psiAlphaHat = psi(0,sum(alpha));
logOmegaTilde = psi(0,alpha) - psiAlphaHat;
logOmegahat = log(h3m.omega);
term_omega = N_supj*(logOmegahat-logOmegaTilde)';

% pi
%N_Eta = h3m.N_Eta;
if ~issyn
for i = 1:K
    N1 = h3m.hmm{i}.N1;
    etai = h3m.hmm{i}.varpar.alpha;
    psiEtaHat = psi(0,sum(etai));
    logPiTilde = psi(0,etai) - psiEtaHat;
    logPihat = log(h3m.hmm{i}.prior);
    term_pi1(i) = N1'*(logPihat(:)-logPiTilde(:));
end

term_pi =sum(term_pi1);

%A

for i = 1:K
    epsilon = h3m.hmm{i}.varpar.epsilon;
    M = h3m.hmm{i}.M;
    for k = 1: size(epsilon,1)
        psiEpsHat = psi(0,sum(epsilon(k,:)));
        logEpsTilde = psi(0,epsilon(k,:)) - psiEpsHat;
        logEpshat = log(h3m.hmm{i}.trans(k,:));
        term_eps1(k) = M(k,:)*(logEpshat-logEpsTilde)';
    end
    term_eps2(i) = sum(term_eps1);
end    

term_eps = sum(term_eps2);

% mu
for i = 1:K
   
    gammai = h3m.hmm{i}.varpar.beta;
    term_mu1(i) = sum(gamma0./gammai);
end
term_mu = -0.5*sum(term_mu1);
        

% sigma
for i = 1:K
    Si = length(h3m.hmm{i}.prior);
    Nj_rho = h3m.hmm{i}.N;
    Wi = h3m.hmm{i}.varpar.W;
    vi = h3m.hmm{i}.varpar.v;
    for k = 1:Si
        t1  = psi(0, 0.5*repmat(vi(k)+1,dim,1) - 0.5*[1:dim]');
        logLambdaTilde = sum(t1) + dim*log(2)  + log(det(Wi(:,:,k)));
        logLambdahat = log(det(vi(k)*Wi(:,:,k)));
        term_W1(k) =0.5* Nj_rho(k)*(logLambdahat - logLambdaTilde);
    end
    term_W2(i) = sum(term_W1);
end

 term_W = sum(term_W2);
 
 else
         
    N_Eta = h3m.N_Eta;

    for i = 1:K
        etai = h3m.varpar.hmm{i}.eta;
        psiEtaHat = psi(0,sum(etai));
        logPiTilde = psi(0,etai) - psiEtaHat;
        logPihat = log(h3m.hmm{i}.prior);
        term_pi1(i) = N_Eta(:,i)'*(logPihat-logPiTilde);
    end

    term_pi =sum(term_pi1);

    %A
    N_Eps = h3m.N_Eps;
    for i = 1:K
        epsilon = h3m.varpar.hmm{i}.epsilon;
        N_Epsi = N_Eps(:,:,i);
        for k = 1: size(epsilon,1)
            psiEpsHat = psi(0,sum(epsilon(k,:)));
            logEpsTilde = psi(0,epsilon(k,:)) - psiEpsHat;
            logEpshat = log(h3m.hmm{i}.trans(k,:));
            term_eps1(k) = N_Epsi (k,:)*(logEpshat-logEpsTilde)';
        end
        term_eps2(i) = sum(term_eps1);
    end    

    term_eps = sum(term_eps2);

    % mu
    for i = 1:K

        gammai = h3m.varpar.hmm{i}.lambda;
        term_mu1(i) = sum(gamma0./gammai);
    end
    term_mu = -0.5*sum(term_mu1);


    % sigma
    for i = 1:K
        Si = length(h3m.hmm{i}.prior);
        Nj_rho = h3m.Nl_j(:,i);
        Wi = h3m.varpar.hmm{i}.W;
        vi = h3m.varpar.hmm{i}.v;
        for k = 1:Si
            t1  = psi(0, 0.5*repmat(vi(k)+1,dim,1) - 0.5*[1:dim]');
            logLambdaTilde = sum(t1) + dim*log(2)  + log(det(Wi(:,:,k)));
            logLambdahat = log(det(inv(h3m.hmm{i}.pdf{k}.cov)));
            term_W1(k) =0.5* Nj_rho(k)*(logLambdahat - logLambdaTilde);
        end
        term_W2(i) = sum(term_W1);
    end

     term_W = sum(term_W2);
 
 end
 
 P_d = 2*(term_omega + term_pi + term_eps + term_mu +  term_W);

 %%         
     
[h3m2] = hmms_to_h3m(h3m.hmm, 'diag');
 
maxN  = max(cellfun(@(x) length(x.prior), H3M.hmm));
maxN2 = max(cellfun(@(x) length(x.prior), h3m2.hmm));


[L_elbo, ~, ~, ~, ~, ~] = ...
          hem_hmm_bwd_fwd_mex(H3M.hmm,h3m2.hmm, T,1, maxN, maxN2);

log_Z = ones(Kb,1) * log(h3m2.omega) + (Ni * ones(Kb,K)) .* L_elbo;
new_LogLikelihood = sum(logtrick(log_Z')');

if dt
    DIC = 2*P_d  - 2*new_LogLikelihood/T;
else
    
    DIC = 2*P_d  - 2*new_LogLikelihood;
end
    

