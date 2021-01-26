function [xL_elbo,nu_1, update_emit_pr, update_emit_mu, update_emit_M, sum_xi] = hem_bwd_fwd(h3m_b,h3m_r,T,vbhemopt)
% used in the vbhem_h3m_c_step+fc.m
% this function is used to compute the second phi
% and not directly output the phi, we output the sythetic statistics
% nu_1, update_emit_pr, update_emit_mu, update_emit_M, sum_xi

% NOTE: THIS H3M_R should be the updated h3m_r and should have the same
% form with H3M_B

Kr = h3m_r.K;
dim = h3m_b.hmm{1, 1}.emit{1, 1}.nin;
canuseMEX = 0;
if (vbhemopt.useMEX)
  if (exist('hem_hmm_bwd_fwd_mex')==3)
    canuseMEX = 1;
    for n=1:length(h3m_b.hmm)
      for j=1:length(h3m_b.hmm{n}.emit)
        if h3m_b.hmm{n}.emit{j}.ncentres ~= 1
          canuseMEX = 0;
          warning('hem_hmm_bwd_fwd_mex: ncentres should be 1');
        end
        %if ~strcmp(h3m_b.hmm{n}.emit{j}.covar_type, 'diag')
        %  canuseMEX = 0;
        %  warning('hem_hmm_bwd_fwd_mex: only diag is supported');
        %end
        if canuseMEX == 0
          break;
        end
      end
      if canuseMEX == 0
        break
      end
    end
    
  else
    canuseMEX = 0;
    warning('hem_hmm_bwd_fwd_mex: MEX implementation not found');
  end
end

if (canuseMEX)
  % get maxN and maxN2
  maxN  = max(cellfun(@(x) length(x.prior), h3m_b.hmm));
  maxN2 = max(cellfun(@(x) length(x.prior), h3m_r.hmm));
  
else
  
  % make cache for faster E-step
  % v0.73 - put into its own function
  h3m_b_hmm_emit_cache = make_h3m_b_cache(h3m_b);
  
  % initialize arrays for E-step
 % L_elbo           = zeros(Kb,Kr);
  nu_1             = cell(Kb,Kr);
  update_emit_pr   = cell(Kb,Kr);
  update_emit_mu   = cell(Kb,Kr);
  update_emit_M    = cell(Kb,Kr);
  sum_xi           = cell(Kb,Kr);
end

smooth =1;
 if (canuseMEX)
    %% MEX implementation
    
    % 2017-11-25: ABC: BUG FIX: n->1, j->1.  just need to check covariance type
    switch(h3m_b.hmm{1}.emit{1}.covar_type)
      case 'diag'
        [xL_elbo, nu_1, update_emit_pr, update_emit_mu, update_emit_M, sum_xi] = ...
          hem_hmm_bwd_fwd_mex(h3m_b.hmm,h3m_r.hmm,T,smooth, maxN, maxN2);

      case 'full'
        % cache logdets and inverse covariances
        logdetCovR = cell(1,Kr);
        invCovR    = cell(1,Kr);
        for j=1:Kr
           % does not support ncentres>1
          logdetCovR{j} = zeros(1,length(h3m_r.hmm{j}.prior));
          invCovR{j}    = zeros(dim, dim, length(h3m_r.hmm{j}.prior));
          for n=1:length(h3m_r.hmm{j}.prior)
            logdetCovR{j}(n)  = log(det(h3m_r.hmm{j}.emit{n}.covars)); 
            invCovR{j}(:,:,n) = inv(h3m_r.hmm{j}.emit{n}.covars);
          end
        end
        
        [xL_elbo, nu_1, update_emit_pr, update_emit_mu, update_emit_M, sum_xi] = ...
          hem_hmm_bwd_fwd_mex(h3m_b.hmm,h3m_r.hmm,T,smooth, maxN, maxN2, logdetCovR, invCovR);
      
      otherwise  
        error('not supported')
    end
    
    % consistency check
    if 0
      err = [];
      err(end+1) = totalerror(xL_elbo, L_elbo);
      err(end+1) = totalerror(xnu_1, nu_1);
      err(end+1) = totalerror(xsum_xi, sum_xi);
      err(end+1) = totalerror(xupdate_emit_pr, update_emit_pr);
      err(end+1) = totalerror(xupdate_emit_mu, update_emit_mu);
      err(end+1) = totalerror(xupdate_emit_M, update_emit_M);
      if any(err>1e-6)
        warning('mismatch');
        keyboard
      end
    end
    
  else
    %% MATLAB implementation
    % loop reduced mixture components
    for j = 1 : Kr
      
      % get this HMM in the reduced mixture
      hmm_r = h3m_r.hmm{j};
      
      % loop base mixture components
      for i = 1 : Kb
        
        % get this HMM in the base mixture
        hmm_b = h3m_b.hmm{i};
        
        % run the Hierarchical bwd and fwd recursions. This will compute:
        % the ELBO on E_M(b)i [log P(y_1:T | M(r)i)]
        % the nu_1^{i,j} (rho)
        % the sum_gamma  [sum_t nu_t^{i,j} (rho,gamma)] sum_m c.. nu..
        % the sum_gamma  [sum_t nu_t^{i,j} (rho,gamma)] sum_m c.. nu.. mu ..
        % the sum_gamma  [sum_t nu_t^{i,j} (rho,gamma)] sum_m c.. nu.. M .. which accounts for the mu mu' and the sigma
        % [the last 3 are not normalized, of course]
        % the sum_t xi_t^{i,j} (rho,sigma)     (t = 2 ... T)
        [xL_elbo(i,j) ...
          nu_1{i,j} ...
          update_emit_pr{i,j} ...
          update_emit_mu{i,j} ...
          update_emit_M{i,j}  ...
          sum_xi{i,j} ...
          ] = hem_hmm_bwd_fwd(hmm_b,hmm_r,T,smooth, h3m_b_hmm_emit_cache{i});
        
        % consistency check
        if 0
          [xL_elbo ...
            xnu_1 ...
            xupdate_emit_pr ...
            xupdate_emit_mu ...
            xupdate_emit_M  ...
            xsum_xi ...
            ] = hem_hmm_bwd_fwd_OLD(hmm_b,hmm_r,T,smooth);
          err = [];
          err(end+1) = abs(xL_elbo-L_elbo(i,j));
          err(end+1) = sum(abs(xnu_1(:) - nu_1{i,j}(:)));
          err(end+1) = sum(abs(xupdate_emit_pr(:) - update_emit_pr{i,j}(:)));
          err(end+1) = sum(abs(xupdate_emit_mu(:) - update_emit_mu{i,j}(:)));
          err(end+1) = sum(abs(xupdate_emit_M(:)  - update_emit_M{i,j}(:)));
          err(end+1) = sum(abs(xsum_xi(:) - sum_xi{i,j}(:)));
          if any(err>1e-6)
            warning('mismatch')
            keyboard;
          end
          
        end
        % end loop base
      end
      
      % end loop reduced
    end
  end