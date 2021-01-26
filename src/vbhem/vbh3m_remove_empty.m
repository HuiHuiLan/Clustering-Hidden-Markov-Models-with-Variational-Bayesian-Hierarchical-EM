function [h3mo, emptyinds] = vbh3m_remove_empty(h3m, thresh,verbose, sortclusters)


if (nargin<2)
  thresh = 1.0;
end

if (nargin<3)
  verbose = 1;
end
if (nargin<4)
  sortclusters= 'f';
end

if isfield(h3m,'NL')
    N = h3m.NL;
elseif isfield(h3m,'Nj')
    N = h3m.Nj;
end

check  = (N < thresh);
zinds  = find(check);
nzinds = find(~check);

h3mo = h3m;

% remove empty hmms and update information
if ~isempty(zinds)
  if (verbose)
    fprintf('removing hmms: %s\n', mat2str(zinds(:)'));
  end
  
  
  h3mo.hmm = h3m.hmm(nzinds);
  h3mo.omega = h3m.omega(nzinds);
  % 2020-9-10: nomorlize it 
  h3mo.omega = h3mo.omega./sum(h3mo.omega);
  h3mo.Z = h3m.Z(:,nzinds);
  if isfield(h3m,'NL')
      h3mo.NL = h3m.NL(:,nzinds);
      
  elseif isfield(h3m,'Nj')
      h3mo.Nj = h3m.Nj(:,nzinds);
  end

  h3mo.L_elbo = h3m.L_elbo(:,nzinds);
  for i = 1:length(nzinds)
    h3mo.label(h3m.label== nzinds(i))=i;
  end

  % update groups
  h3mo.groups = h3m.groups(1,nzinds);
  h3mo.group_size = h3m.group_size(1,nzinds);

  emptyinds = zinds;
else
  % do nothing
  emptyinds = [];
  
end

% remove empty states
hmms= h3mo.hmm;

for i=1:length(hmms)
    [hmms{i}, zi] = vbhmm_remove_empty(hmms{i}, 0, 1e-3);
    if ~isempty(zi)
      if (1)
        fprintf('%d: removed states', i);
        fprintf(' %d', zi);
        fprintf('; ');
      end
    end

end
h3mo.hmm = hmms;

% make the most possible ROI to the first ROI
 
for j=1:length(h3mo.hmm)
    hmm = h3mo.hmm{j};
    h3mo.hmm{j} = vbhmm_standardize(hmm, sortclusters);
end
