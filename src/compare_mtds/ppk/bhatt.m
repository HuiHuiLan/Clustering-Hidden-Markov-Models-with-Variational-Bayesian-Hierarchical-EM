function [ba] = bhatt(mu1,mu2,cov1,cov2)
%%%% Code to compute the kernel between two Gaussian emission HMMs

%   function [ba] = bhatt(mu1,mu2,cov1,cov2)
%       Calculates the bhattacharya affinity for two distributions
%       mu1, mu2, cov1 and cov2 are the means and covariances
%           mu1 and mu2 are column vectors, cov1 and cov2 are square
%           matrices
%
%   Yingbo Song (yingbo@cs.columbia.edu)

  [n m] = size(cov1);

% add a small regularizer to the diagonal, ensure covm is full rank
cov1 = cov1 + 1e-5*trace(cov1)*eye(n);
cov2 = cov2 + 1e-5*trace(cov2)*eye(n);

iC1 = inv(cov1);
iC2 = inv(cov2);
Cd = inv(iC1 + iC2);
Md = iC1*mu1 + iC2*mu2;

% calculate the affinity
rho = 0.5;     % exponential, should be 0.5 for Bhattacharyya affinity

normalizer = (2*pi)^((1-2*rho)*(n/2)) * rho^(-n/2);
normalizer = normalizer * det(cov1)^(-rho/2) * det(cov2)^(-rho/2) * sqrt(det(Cd));
ba = normalizer * exp((-rho/2)*(mu1'*(iC1)*mu1 + mu2'*(iC2)*mu2 - Md'*(Cd)*Md));

% end of function

