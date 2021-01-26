function [h3m]= ppk_sc(hmms,K,seed)
% This function is to implement the PPK-SC clustering algorithm
% hmms: all the hmms want to be clustered? is 1*N cell
% h3m: the clustering results
% This code follow the ppk-sc paper

if nargin<3
    seed = [];
end

N = length(hmms);

beta = 0.5;
T = 10;

% compute the Gram matrix A
A = zeros(N,N);
for i =1:N
    for j =1:N
        A(i,j) = elkernel(hmms{i},hmms{j},T,beta);
    end
end


[ C, C_centres, ~,U ] = SpectralClustering(A, K, 3, seed);
% C_centres is k*dim
% REFORM C to get assignment
Z = full(C);
indx = zeros(N,1);
for k =1:K
    indx(Z(:,k)==1) = k;
end


% find the center
%One simple extension of PPK-SC to obtain a HMM cluster center is to select the input HMM 
%that the spectral clustering algorithm maps closest to the spectral clustering center.
ind_center = zeros(K,1);
myfun=@(x,y) sqrt(sum((x-y).^2,2));
for k = 1:K    
    indxk = find(indx==k);
    tmp = U(indxk,:);
    tmp2 = myfun(C_centres(k,:),tmp);
    ind_center(k) = indxk(find(tmp2== min(tmp2),1));   
end

h3m = {};
hmms_center = cell(1,K);
weitgt = zeros(K,1);

for k =1:K
    group{k} = find(indx==k);
    group_size(k) = sum(indx==k);
    weitgt(k) = group_size(k)/N;
end

    
h3m.group_size = group_size;
h3m.group = group;
h3m.weight = weitgt;
h3m.label = indx;
h3m.Z = Z;
for k =1:K
    hmms_center{k} = hmms{ind_center(k)};
end
h3m.hmms = hmms_center;



% if 0
% 
% % creat D
% D = diag(sum(A,2));
% L = D^(-0.5)*A*D^(-0.5);
% [V,Lambda]=eig(L);
% 
% %find(diag(Lambda),K);
% 
% [sortedX, sortedInds] = sort(diag(Lambda),'descend');
% topK = sortedInds(1:K);
% 
% X = V(:,topK);
% % row normalize  
% if K>1
% X = X./sum(X,2);
% end
% % X N*p
% [indx,init_center] = kmeans(X, K,'MaxIter', 200);
% 






