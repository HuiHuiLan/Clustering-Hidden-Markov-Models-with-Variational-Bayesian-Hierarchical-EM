function [ C_new,C_cntres, L, U ] = SpectralClustering( W, k, Type ,seed)
%SPECTRALCLUSTERING Executes spectral clustering algorithm
%   Executes the spectral clustering algorithm defined by
%   Type on the adjacency matrix W and returns the k cluster
%   indicator vectors as columns in C.
%   If L and U are also called, the (normalized) Laplacian and
%   eigenvectors will also be returned.
%
%   'W' - Adjacency matrix, needs to be square
%   'k' - Number of clusters to look for
%   'Type' - Defines the type of spectral clustering algorithm
%            that should be used. Choices are:
%      1 - Unnormalized
%      2 - Normalized according to Shi and Malik (2000)
%      3 - Normalized according to Jordan and Weiss (2002)
%
%   References:
%   - Ulrike von Luxburg, "A Tutorial on Spectral Clustering", 
%     Statistics and Computing 17 (4), 2007
%
%   Author: Ingo Buerk
%   Year  : 2011/2012
%   Bachelor Thesis

% calculate degree matrix
if nargin<4
    seed = [];
end
degs = sum(W, 2);
D    = sparse(1:size(W, 1), 1:size(W, 2), degs);

% compute unnormalized Laplacian
%L = D - W;
L = W;
% compute normalized Laplacian if needed
switch Type
    case 2
        % avoid dividing by zero
        degs(degs == 0) = eps;
        % calculate inverse of D
        D = spdiags(1./degs, 0, size(D, 1), size(D, 2));
        
        % calculate normalized Laplacian
        L = D * L;
    case 3
        % avoid dividing by zero
        degs(degs == 0) = eps;
        % calculate D^(-1/2)
        D = spdiags(1./(degs.^0.5), 0, size(D, 1), size(D, 2));
        
        % calculate normalized Laplacian
        L = D * L * D;
end

% compute the eigenvectors corresponding to the k smallest
% eigenvalues
% diff   = eps;
% [U, ~] = eigs(L, k, diff);

% compute the eigenvectors corresponding to the k largest
% eigenvalues
% for the case that U is complex

%L = (L+L')/2;
[U, ~] = eigs(L, k);

if ~isreal(sum(U(:)))
    L = (L+L')/2;
    [U, ~] = eigs(L, k);
end
% in case of the Jordan-Weiss algorithm, we need to normalize
% the eigenvectors row-wise
if Type == 3
    U = bsxfun(@rdivide, U, sqrt(sum(U.^2, 2)));
    U(isnan(U)) = 0;
end

% now use the k-means algorithm to cluster U row-wise
% C will be a n-by-1 matrix containing the cluster number for
% each data point
% U
% size(U)
if ~isempty(seed)
    rng(seed, 'twister');
end
      
try
    
[C,C_cntres] = kmeans(U, k, 'start', 'cluster', ...
                 'EmptyAction', 'singleton', 'Display', 'off', 'MaxIter', 200);
catch ME
    [C,C_cntres] = kmeans(U, k,  ...
                 'EmptyAction', 'singleton', 'Display', 'off', 'MaxIter', 200);
end
             
% now convert C to a n-by-k matrix containing the k indicator
% vectors as columns
C_new = sparse(1:size(D, 1), C, 1);

end