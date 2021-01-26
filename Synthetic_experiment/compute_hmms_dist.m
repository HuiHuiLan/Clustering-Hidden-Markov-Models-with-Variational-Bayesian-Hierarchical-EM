function [Dists] = compute_hmms_dist(centHMM,indHMMs,indData,Groups,type,cent_ids)
% compute the distance between hmms with the cluster center
% centHMM:can be cell or struct
% HMMs : can be cell of cell or just cell
% indData : the data conrespponding to the HMMs
% Group: member of each groups
% type: vb == vbhem
%       vh : vhem
%       sc : ppk-sc
%       cf : ccfd
if nargin <6
    cent_ids =[];
end

Ncenters = length(centHMM);
Cendata = cell(Ncenters,1);

switch type
    
    case {'vb','vh','dic'}
        for j =1: Ncenters
            group_ind = Groups{j};
            tmpdata = indData(group_ind(:),1);
            Cendata{j} = cat(1,tmpdata{:});
        end
        
    case {'sc'}
        
        for j =1: Ncenters
            
            subidx = centHMM{j}.vbopt.verbose_prefix;
            itself_idx = str2num(subidx(isstrprop(subidx,'digit')));
            tmpdata = indData(itself_idx,1);
            Cendata{j} = cat(1,tmpdata{:});
        end
    case {'ccfd'}
        
        for j =1: Ncenters
            
            itself_idx = cent_ids(j);
            tmpdata = indData(itself_idx,1);
            Cendata{j} = cat(1,tmpdata{:});
        end
end


Dists = cell(Ncenters,1);

for j =1: Ncenters
    centhmm = centHMM{j};
    cendata = Cendata{j};

    tmpHMMs = indHMMs(Groups{j});
    tmpdata = indData(Groups{j});
    Mdist = zeros(length(tmpHMMs),1);

    for i =1:length(tmpHMMs)     
        hmm_i = tmpHMMs{i};
        data_i = tmpdata{i};
        Mdist(i,1) = 0.5*(vbhmm_kld(centhmm, hmm_i, cendata) + vbhmm_kld(hmm_i, centhmm, data_i));
    
    end
    Dists{j} = Mdist;
    
end
