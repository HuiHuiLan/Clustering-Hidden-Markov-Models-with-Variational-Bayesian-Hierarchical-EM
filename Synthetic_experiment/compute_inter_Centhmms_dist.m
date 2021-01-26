function [inter_dists_vct,inter_Dists] = compute_inter_Centhmms_dist(centHMM,indData,Groups,type,cent_ids)
% compute the distance (SKLD) between hmms cluster centers

if nargin <5
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


inter_Dists = zeros(Ncenters,Ncenters);
t=1;
for i =1: Ncenters
    centhmm1 = centHMM{i};
    cendata1 = Cendata{i};
    
    for j = (i+1):Ncenters
        centhmm2 = centHMM{j};
        cendata2 = Cendata{j};
        inter_Dists(i,j) = 0.5*(vbhmm_kld(centhmm1, centhmm2, cendata1) + vbhmm_kld(centhmm2, centhmm1, cendata2));
        inter_dists_vct(t) = inter_Dists(i,j);
        t = t+1;
    end
    
end
