function [K] = elkernel(hmm1,hmm2,T,rho)
%   function [K] = elkernel(hmm1,hmm2,T)
%       Tony's iterative PPK
%   NOT in logspace

if ( sum(hmm1.prior) < 0 )
  error('elkernel: The hmm parameters cannot be in log-space!'); 
end

prior1 = hmm1.prior;
prior2 = hmm2.prior;
transmat1 = hmm1.trans;
transmat2 = hmm2.trans;
M = length(prior1);
N = length(prior2);

pad = 0.45;  % i don't know what is this!

pot = zeros(M,N);
for i=1:M
  for j=1:N
    pot(i,j) = bhatt(hmm1.pdf{i}.mean',hmm2.pdf{j}.mean',...
    hmm1.pdf{i}.cov + pad, hmm2.pdf{j}.cov + pad);
  end
end


K = 0;
if (T==1)
    for i=1:M
        for j=1:N
            K=K+prior1(i)*prior2(j)*pot(i,j);
        end
    end
else
    
    sep1 = zeros(M,N);
    for i=1:M
        for j=1:N
            sep1 = sep1 + (prior1(i)*prior2(j))^rho*pot(i,j)*(transmat1(i,:).^rho)'*(transmat2(j,:).^rho);
        end
    end
    
    for t=2:T
        sep2 = zeros(M,N);
        for i=1:M
            for j=1:N
                sep2 = sep2 + sep1(i,j)*pot(i,j)*(transmat1(i,:)'*transmat2(j,:)).^rho;
            end
        end
        sep1 = sep2;
    end
    K = sum(sum(sep2.*pot));
end