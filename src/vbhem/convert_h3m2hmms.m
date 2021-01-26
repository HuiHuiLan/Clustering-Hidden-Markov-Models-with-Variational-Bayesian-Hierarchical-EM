function hmms_new=convert_h3m2hmms(h3m)
% covert the h3m from vbh3m to hmms, then use hem to reduce hmms

hmms1 = h3m.hmm;

len_hmms = length(hmms1);
hmms = cell(1,len_hmms);
for i = 1:len_hmms
    hmms{1, i}.prior = hmms1{1, i}.prior;
    hmms{1, i}.trans = hmms1{1, i}.trans;
    hmms{1, i}.pdf = hmms1{1, i}.pdf';
    if isfield(hmms1{1, i},'gamma')
    hmms{1, i}.gamma = hmms1{1, i}.gamma;
    end
    hmms{1, i}.M = h3m.NJ_ror(:,:,i);
    hmms{1, i}.N1 = h3m.NJ_rho1(:,i);
    hmms{1, i}.N = h3m.NJ_rho(i,:)';
    hmms{1, i}.varpar.alpha = h3m.varpar.alpha{1,i};
    hmms{1, i}.varpar.beta = h3m.varpar.beta(:,:,i);
    hmms{1, i}.varpar.epsilon = h3m.varpar.epsilon{:,i};
    hmms{1, i}.varpar.v = h3m.varpar.v(:,:,i);
    hmms{1, i}.varpar.m = h3m.varpar.m{1,i};
    hmms{1, i}.varpar.W = h3m.varpar.W{1,i};
end

hmms_new = hmms;
    
