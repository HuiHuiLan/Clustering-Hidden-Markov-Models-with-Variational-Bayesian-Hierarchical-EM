function [CCFD_HMMs] = myccfd(hmms,data,slope)
% % slope is k in the paper

if nargin<3
    slope = 3;
end


ND = length(hmms);
N = ND*(ND-1)/2;
k = 1;


dist = zeros(ND,ND);
pur =[];

for i =1:ND
    hmm1 = hmms{i};
    data1 = data{i};
    for j =(i+1):ND
        
        hmm2 = hmms{j};
        data2 = data{j};
        
        dist(i,j) = 0.5*(vbhmm_kld(hmm1, hmm2, data1) + vbhmm_kld(hmm2, hmm1, data2));
        dist(j,i) = 0.5*(vbhmm_kld(hmm1, hmm2, data1) + vbhmm_kld(hmm2, hmm1, data2));
        pur(k) = dist(i,j);
        k = k+1;
    end
end

% 
if 0
percent=2.0;

position=round(N*percent/100);
sda=sort(pur(:));
dc=sda(position);
end

% given a initial dc
percent = 10;
r = 3;
cofr = [-1,0,1];
fitness=[];
dc=[];
icl={};
cl={};


while(r>0)
    
    %p0 = percent;
    fitness = zeros(1,3);
    for i = 1:3
        p0 = percent + r*cofr(i);
        try
        [fitness(i),dc(i),icl{i},cl{i}] = CCFD(dist,p0,pur,slope);
        catch ME
            fitness(i)=-realmax;
        end
    end
    
    if sum(fitness==max(fitness))==3
        fitidx=3;
       %[maxfit,fitidx] = max(fitness);
    else
        [maxfit,fitidx] = max(fitness);
    end
    
    percent = percent + r*cofr(fitidx);
    r = r-0.5;
end

dc_star = dc(fitidx);

[fitness_star,dc_star,icl_star,cl_star,rho_star,delta_star,halo_star] = CCFD(dist,p0,pur,slope,dc_star);


clster_center = icl{fitidx};
label = cl{fitidx};
NCLUST = length(clster_center);


ccfd_result = struct;
ccfd_result.fitness = fitness_star;
ccfd_result.percent = percent;
ccfd_result.dc = dc_star;
ccfd_result.icl =clster_center;
ccfd_result.cl = label ;
ccfd_result.rho = rho_star;
ccfd_result.delta = delta_star;
ccfd_result.NCLUST = NCLUST;
ccfd_result.halo = halo_star;
ccfd_result.dist = dist;
ccfd_result.k = slope;


CCFD_HMMs = struct;
CCFD_HMMs.ccfd_result = ccfd_result;

for i = 1:NCLUST
    CCFD_HMMs.hmms{i} = hmms{clster_center(i)};
end
CCFD_HMMs.label = label;
groups={};
group_size = zeros(NCLUST,1);
for i =1:NCLUST
    groups{i} = find(label==i);
    group_size(i) = length(groups{i});
end

CCFD_HMMs.groups = groups;
CCFD_HMMs.weight = group_size./sum(group_size);
