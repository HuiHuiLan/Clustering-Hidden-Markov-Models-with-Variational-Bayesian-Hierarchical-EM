%% The journal synthetic experiments
%% experiment 1
%% load data
%clear 

% load_results
load('data.mat')
load('HMMs.mat')
load('CCFD.mat')
load('VHHMMs.mat')
%load('APPK_h3m.mat')
load('PPK.mat')
load('VBgroup_hmms.mat')

%% exp1
K = 1:6;
S = 1:5;

true_label=[];
for i =1:2
    true_label = [true_label,i*ones(1,20)];
end

% %% exp2
% K = 1:10;
% S = 1:10;
% 
% true_label=[];
% for i =1:num_hmm
%     true_label = [true_label,i*ones(1,20)];
% end

%% evaluate synthetic data

cpt_dist = 0;
%% VHEM

[VHEM_results] = evaluate_vbhem_jounarl(vh_hmms,true_label,'vhem',K,S,num_hmm,num_sta,HMMs,Adata,[],cpt_dist);

%save('VHEM_results.mat')

%% VHAIC

[VHAIC_results] = evaluate_vbhem_jounarl(vh_hmms,true_label,'vhaic',K,S,num_hmm,num_sta,HMMs,Adata,[],0);
%save('VHAIC_results.mat','VHAIC_results')

%% VHBIC

[VHBIC_results] = evaluate_vbhem_jounarl(vh_hmms,true_label,'vhbic',K,S,num_hmm,num_sta,HMMs,Adata,[],0);


%% PPK-SC AIC
sum(isnan(PPK_LL(:)))

[SCAIC_results] = evaluate_vbhem_jounarl(APPK_h3m,true_label,'scaic',K,S,num_hmm,num_sta,HMMs,Adata,PPK_LL,0);


%% PPK-SC BIC
sum(isnan(PPK_LL(:)))

[SCBIC_results] = evaluate_vbhem_jounarl(APPK_h3m,true_label,'scbic',K,S,num_hmm,num_sta,HMMs,Adata,PPK_LL,0);

%% CCFD
[CCFD_results] = evaluate_vbhem_jounarl(AllCCFD_HMMs,true_label,'ccfd',K,S,num_hmm,num_sta,HMMs,Adata,[],0);

%% VBHEM
[VBHEM_results] = evaluate_vbhem_jounarl(group_vbhmms,true_label,'vb',K,S,num_hmm,num_sta,HMMs,Adata,[],0);
%save('VBHEM_K5S3_results.mat','VBHEM_results')

%% compute DIC

[DIC_results] = evaluate_vbhem_jounarl(group_vbhmms,true_label,'dic',K,S,num_hmm,num_sta,HMMs,Adata,[],0);

%save('DIC_results.mat')


%%   OLD CODE %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% BIC
if 0
if 0
    
    %%% old code
Nv = vh_hmms{1, 1}{2, 2}.hemopt.Nv;
T = vh_hmms{1, 1}{2, 2}.hemopt.tau;
hemopt =  vh_hmms{1, 1}{2, 2}.hemopt;

N_bic = Nv*Kb*T;

for it = 1:iter
    
    hmms_it = vh_hmms{it};
    for k = 1:length(K)
        kk = K(k);
        for s = 1:length(S)
            
            ss = S(s);
            hmms_ks = hmms_it{kk,ss};
            Num_pars = (kk-1) + kk*((ss-1) + ss*(ss-1) + ss*2*dim);
            omega_ks = sum(hmms_ks.Z,1)./Kb;
            log_ests = sum(sum(hmms_ks.Z.*( log(omega_ks).*ones(Kb,1) -log(hmms_ks.Z+1e-50) + hemopt.Nv* hmms_ks.L_elbo1)));
            BIC_it(k,s) = log(N_bic)*Num_pars-2*log_ests;
 
        end
    end
    
    BIC{it} = BIC_it;
end


% find the model with smallest BIC, and determine the K &S
is_K_right_vh = zeros(iter,1);
is_K_big_vh = zeros(iter,1);
is_K_less_vh = zeros(iter,1);

rate_S_right_vh = zeros(iter,1);
rate_S_big_vh =zeros(iter,1);
rate_S_less_vh =zeros(iter,1);

RI_vhbic=zeros(iter,1);
Puri_vhbic=zeros(iter,1);

% 

for it = 1:iter
    
    BIC_it = BIC{it};  

    [min_indk,min_inds] = find(BIC_it==min(BIC_it(:)));

        
    hmms_it = vh_hmms{it};        
    hmms_ks = hmms_it{min_indk,min_inds};
      
%     vh_weight = hmms_ks.weight;
%     K_vh= sum(vh_weight > (1/Kb)); 
    K_vh= sum(hmms_ks.group_size~=0); 
    
    len_hmm_s = length(hmms_ks.hmms);
    S_vhj = zeros(len_hmm_s,1);
    for j = 1:len_hmm_s
        S_vhj(j)= sum( hmms_ks.hmms{1, j}.prior > (1/Kb));
    end
    
    
    
    rate_S_right_vh(it) = sum(S_vhj==num_sta)/len_hmm_s;
    rate_S_big_vh(it) = sum(S_vhj<num_sta)/len_hmm_s;
    rate_S_less_vh(it) = sum(S_vhj>num_sta)/len_hmm_s;
    
    is_K_right_vh(it) = K_vh==num_hmm;
    is_K_big_vh(it) = K_vh>num_hmm;
    is_K_less_vh(it) = K_vh<num_hmm;
    
            
    vhbic_label = hmms_ks.label;
    RI_vhbic(it)=valid_RandIndex(true_label,vhbic_label);       
    Puri_vhbic(it)=Purity(true_label,vhbic_label); 
    
   % [Dists] = compute_hmms_dist(centHMM,indHMMs,indData,Groups,type);
end


mean(RI_vhbic)
std(RI_vhbic)

mean(Puri_vhbic)
std(Puri_vhbic)

mean(is_K_right_vh)
std(is_K_right_vh)

mean(rate_S_right_vh)
std(rate_S_right_vh)

mean(is_K_big_vh)
std(is_K_big_vh)

mean(rate_S_big_vh)
std(rate_S_big_vh)


mean(is_K_less_vh)
std(is_K_less_vh)

mean(rate_S_less_vh)
std(rate_S_less_vh)

end

%% AIC

if 0

Kb =40;
%N_AIC = Kb;

dim =2;
AIC = [];
hemopt = vh_hmms{1, 1}{2, 2}.hemopt;

iter = length(vh_hmms);

for it = 1:iter   
     hmms_it = vh_hmms{(it)}; 
     AIC_it =[];
    for k = 1:length(K)
        for s = 1:length(S)
            
            hmms_ks = hmms_it{k,s};
            Num_pars = k*s*(s+2*dim)-1;
            omega_ks = sum(hmms_ks.Z,1)./Kb;
            log_ests = sum(sum(hmms_ks.Z.*( log(omega_ks).*ones(Kb,1) -log(hmms_ks.Z+1e-50) + hemopt.Nv* hmms_ks.L_elbo1)));
            AIC_it(k,s) = 2*Num_pars-2*log_ests;
        end
    end
    
    AIC{it} = AIC_it;
end


%plot( AIC_it(k,:))


is_K_right_vh = zeros(iter,1);
is_K_big_vh = zeros(iter,1);
is_K_less_vh = zeros(iter,1);

rate_S_right_vh = zeros(iter,1);
rate_S_big_vh =zeros(iter,1);
rate_S_less_vh =zeros(iter,1);

RI_vhaic=zeros(iter,1);
Puri_vhaic=zeros(iter,1);

for it = 1:iter
    
    AIC_it = AIC{it};  

    [min_indk,min_inds] = find(AIC_it==min(AIC_it(:)));
  %  KK(it) = min_indk;
        
    hmms_it = vh_hmms{it};        
    hmms_ks = hmms_it{min_indk,min_inds};
      
   % vh_weight = hmms_ks.weight;
    K_vh= sum(hmms_ks.group_size~=0); 
    %K_vh= sum(vh_weight > (1/Kb)); 
    
    len_hmm_s = length(hmms_ks.hmms);
    S_vhj = zeros(len_hmm_s,1);
   % s_total = s_total+length(hmms_ks.hmms);
    for j = 1:len_hmm_s

        S_vhj(j)= sum( hmms_ks.hmms{1, j}.prior > (1/Kb));
    end

    %S_all = [S_all;S_vhj];
    rate_S_right_vh(it) = sum(S_vhj==num_sta)/len_hmm_s;
    rate_S_big_vh(it) = sum(S_vhj<num_sta)/len_hmm_s;
    rate_S_less_vh(it) = sum(S_vhj>num_sta)/len_hmm_s;
    
    is_K_right_vh(it) = K_vh==num_hmm;
    is_K_big_vh(it) = K_vh>num_hmm;
    is_K_less_vh(it) = K_vh<num_hmm;
   
                
    vhaic_label = hmms_ks.label';
    RI_vhaic(it)=valid_RandIndex(true_label,vhaic_label);    
    Puri_vhaic(it)=Purity(true_label,vhaic_label);       

    
end


mean(RI_vhaic)
std(RI_vhaic)

mean(Puri_vhaic)
std(Puri_vhaic)

mean(is_K_right_vh)
std(is_K_right_vh)

mean(rate_S_right_vh)
std(rate_S_right_vh)

mean(is_K_big_vh)
std(is_K_big_vh)

mean(rate_S_big_vh)
std(rate_S_big_vh)


mean(is_K_less_vh)
std(is_K_less_vh)

mean(rate_S_less_vh)
std(rate_S_less_vh)
end

%% VBHEM


if 0
N_iters = length(group_vbhmms);

start_iter = 1;
iter = N_iters;


K_cls = zeros(iter,1);
isK_right = zeros(iter,1);
RI = zeros(iter,1);
Puri_vb= zeros(iter,1);
S_all =[];

for i = start_iter:iter
    [h3mo, emptyinds] = vbh3m_remove_empty(group_vbhmms{i});
    K_cls(i) = length(h3mo.group_size)-sum(h3mo.group_size==0);
    hmm_learn = h3mo.hmm;
    for j = 1:length(hmm_learn)
        S_temp(j) = length(h3mo.hmm{1, j}.prior);
    end
    S_all = [S_all,S_temp];   
        
    RI(i) =valid_RandIndex(true_label,h3mo.label);
    Puri_vb(i)=Purity(true_label,h3mo.label);   
    
    isK_right(i) = K_cls(i)==num_hmm;
end

for it =1:iter
    
    tmpgrouphmm = vbh3m_remove_empty(group_vbhmms{it});
    centHMM = tmpgrouphmm.hmm;
    indHMMs = HMMs{it};
    Groups = tmpgrouphmm.groups;
    indata = Adata{it};   
    [Mdist] = compute_hmms_dist(centHMM,indHMMs,indata,Groups,'vb');
    all_vb_dist{it} = Mdist;
end

% mean(RI(start_iter:iter))
% mean(Puri_vb)
% sum(isK_right(start_iter:iter))/N_iters
% sum(S_all(start_iter:iter)==num_sta)/length(S_all(start_iter:iter))

mean(RI)
std(RI)

mean(Puri_vb)
std(Puri_vb)

mean(isK_right)
std(isK_right)


mean(S_all==num_sta)
std(S_all==num_sta)

mean(K_cls>num_hmm)
std(K_cls>num_hmm)

mean(S_all>num_sta)
std(S_all>num_sta)

mean(K_cls<num_hmm)
std(K_cls<num_hmm)

mean(S_all<num_sta)
std(S_all<num_sta)

end

%% DIC

if 0
    %%%% old code %%%%
RI_dic =[];
Puri_dic =[];
isK_right_dic=[];
isK_large=[];
isK_less=[];

iter = length(group_vbhmms);

for i =1:iter
    vbh3m = group_vbhmms{1, i};
    hmms = HMMs{i};
    P_d=[];
    DIC = [];
    for k =1:6
        for s = 1:5
            vbh3mj = vbh3m.model_all{s, k};
            [P_d(k,s),DIC(k,s)] = myDIC(hmms, vbh3mj,group_vbhmms{1, 1}.vbhembopt.tau);
        end
    end
 
    p_D{i} = P_d;
    DICs{i} = DIC;
    [min_indk,min_inds] = find(DIC==min(DIC(:)));
    
    RI_dic(i) =valid_RandIndex(true_label, vbh3m.model_all{min_inds, min_indk}.label); 
    Puri_dic(i) =Purity(true_label, vbh3m.model_all{min_inds, min_indk}.label);    

    K_is(i) = min_indk;
    isK_right(i) = min_indk==2;
    isK_large(i) = min_indk>2;
    isK_less(i) = min_indk<2;
    isS_right(i) = min_inds==2;
    isS_large(i) = min_inds>2;
    isS_less(i) = min_inds<2;

end


mean(RI_dic)
std(RI_dic)


mean(Puri_dic)
std(Puri_dic)

mean(isK_right)
std(isK_right)

mean(isS_right)
std(isS_right)

sum(isK_large)/iter
std(isK_large)

sum(isS_large)/iter
std(isS_large)

sum(isK_less)/iter
std(isK_less)

sum(isS_less)/iter
std(isS_less)


DIC = DICs{40};
a = DIC(2,:)
b= DIC(:,2)'

plot(DIC(2,:))
plot(DIC(:,2))

end



if 0
RI_ccfd = zeros(1,iter);
Puri_ccfd = zeros(1,iter);

is_K_right_ccfd = zeros(iter,1);
is_K_big_ccfd = zeros(iter,1);
is_K_less_ccfd = zeros(iter,1);


for i =1:iter 
    try
    ccfd_label = AllCCFD_HMMs{i}.label;
    catch
        continue
    end
    
    
    RI_ccfd(i) =valid_RandIndex(true_label, ccfd_label); 
    Puri_ccfd(i) =Purity(true_label,ccfd_label);  
    K_ccfd(i) = length(AllCCFD_HMMs{i}.ccfd_result.icl);
   

end

mean(RI_ccfd)
std(RI_ccfd)

mean(Puri_ccfd)
std(Puri_ccfd)

mean(K_ccfd==2)
std(K_ccfd==2)

mean(K_ccfd>2)
std(K_ccfd>2)

mean(K_ccfd<2)
std(K_ccfd<2)
end


%% vhem

true_label = [];
for i =1:2
    true_label = [true_label,i*ones(1,20)];
end



for it = 1:iter
    
    hmms_it = vh_hmms{(it)}; 
    RI_vh = zeros(length(K),length(S));
    Purity0 = zeros(length(K),length(S));
    for k = 1:length(K)
        for s = 1:length(S)
            
            hmms_ks = hmms_it{k,s};
            vh_label = hmms_ks.label;
            RI_vh(k,s)=valid_RandIndex(true_label,vh_label);
            [Purity0(k,s)] =Purity(true_label,vh_label);

        end
    end
    
    RI_VH{it} = RI_vh;
    Purity_VH{it} = Purity0;
end




temp = cat(1,RI_VH{:});
mean_RI_VH = mean(temp(:))
std(temp(:))

temp = cat(1,Purity_VH{:});
mean_Rur_VH = mean(temp(:))
std(temp(:))

Right_K_vh = [];
Right_S_vh = [];
Big_K_vh = [];
Big_S_vh = [];
Less_K_vh = [];
Less_S_vh = [];
Sall =[];

for it = 1:iter
    
    hmms_it = vh_hmms{it}; 
    right_K_vh =[];
    big_K_vh =[];
    less_K_vh=[];
    right_S_vh =[];
    big_S_vh =[];
    less_S_vh=[];
    for k = 1:length(K)
        for s = 1:length(S)
            
            hmms_ks = hmms_it{k,s};
            %vh_weight = sum(hmms_ks.Z,1)/Kb;
            K_vh= sum(hmms_ks.group_size~=0); 

            %K_vh= sum(vh_weight > (1/Kb));   %K for this trail
            right_K_vh(k,s) = K_vh==num_hmm;
            big_K_vh (k,s) = K_vh>num_hmm;
            less_K_vh (k,s) = K_vh<num_hmm;
            S_vhj = zeros(length(hmms_ks.hmms),1);
            for j = 1:length(hmms_ks.hmms)
                
                S_vhj(j)= sum( hmms_ks.hmms{1, j}.prior > (1/Kb));
            end
            
            Sall =[Sall,S_vhj'];
            right_S_vh(k,s)= sum(S_vhj == num_sta)/length(hmms_ks.hmms);  % the rate of S is right for this trail
            big_S_vh(k,s)= sum(S_vhj > num_sta)/length(hmms_ks.hmms);     
            less_S_vh(k,s)= sum(S_vhj < num_sta)/length(hmms_ks.hmms);     
        
        end
    end
    
    Right_K_vh = [Right_K_vh,right_K_vh];
    Right_S_vh = [Right_S_vh,right_S_vh];
    
    Big_K_vh = [Big_K_vh,big_K_vh];
    Big_S_vh = [Big_S_vh,big_S_vh];
    
    Less_K_vh = [Less_K_vh,less_K_vh];
    Less_S_vh = [Less_S_vh,less_S_vh];
end

rate_K_right_VH = mean(Right_K_vh(:))
std(Right_K_vh(:))
%mean(Sall ==num_sta)
rate_S_right_VH = mean(Right_S_vh(:))
std((Right_S_vh(:)))

rate_K_big_VH = mean(Big_K_vh(:))
std(Big_K_vh(:))

rate_S_big_VH = mean(Big_S_vh(:))
std(Big_S_vh(:))

%mean(Sall >num_sta)
rate_K_less_VH = mean(Less_K_vh(:))
std((Less_K_vh(:)))

rate_S_less_VH = mean(Less_S_vh(:))
std((Less_S_vh(:)))

end