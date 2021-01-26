function [Output] = evaluate_vbhem_jounarl(Group_hmms,true_label,method,K,S,num_hmm,num_sta,HMMs,Datas,PPK_LL,cpt_dist,div_T)
%% evaluate the reuslts from the experiment in vbhem journal
% group_vbhmms : the cluster results
% method: vb == vbhem
%         dic : DIC
%         vh : vhem
%         sc : ppk-sc
%         cf : ccfd
% HMMs : the input be clustered
% Datas : the data be used to learn HMMs

iter = length(Group_hmms);
Kb = length(HMMs{1});
dim = length(HMMs{1, 1}{1, 1}.vbopt.mu0);
len_K = length(K);
len_S = length(S);

if nargin <12
    div_T=0;
end

if nargin <11
    cpt_dist=0;
end
    

if (nargin <10) && (strncmp(method,'sc',2))

    PPK_LL = zeros(len_K,len_S,iter);
    APPK_h3m = Group_hmms;
    
    for it = 1:iter
        fprintf('=== LOOP %d, %d ===\n',  it);
        data_test = Datas{it};
        PPK_h3m = APPK_h3m{it};
        
        for kk = 1:len_K
            Kgroup = K(kk);
            
            for ss = 1:len_S
                
                tmph3m = PPK_h3m{kk,ss};
                
                Group_data = cat(1,data_test{:});
                loglik_hmm = zeros(length(Group_data),Kgroup);
                for k = 1:Kgroup
                    [tmphmms] = vbhmm_remove_empty(tmph3m.hmms{k}, 0, 1e-3);
                    
                    [loglik_hmm(:,k)]= vbhmm_ll(tmphmms,Group_data);
                end
                
                PPK_LL(kk,ss,it)  = sum(logtrick(log(tmph3m.weight) + loglik_hmm'));
            end
        end
    end
    
end

     
RI=zeros(iter,1);
Puri=zeros(iter,1);
DI=zeros(iter,1);

if (strcmp(method,'vhem'))
    K_is = cell(iter,1);
else
    K_is = zeros(iter,1);
end

is_K_right = zeros(iter,1);
is_K_big = zeros(iter,1);
is_K_less = zeros(iter,1);


if (len_S>1)&&(~strcmp(method,'ccfd'))
    S_is= cell(iter,1);
    is_S_right = zeros(iter,1);
    is_S_big =zeros(iter,1);
    is_S_less =zeros(iter,1);    
end


%% compute DIC
switch method
    
    case 'vb'
        
        group_vbhmms = Group_hmms;
        
        for i = 1:iter
            
            tmph3m = vbh3m_remove_empty(group_vbhmms{i});
            K_select = length(tmph3m.omega);
            S_select = cellfun(@(x) length(x.prior),tmph3m.hmm);
            
            LABEL = tmph3m.label;
            
            RI(i) =valid_RandIndex(true_label, LABEL);
            Puri(i) =Purity(true_label, LABEL);
            
            is_K_right(i) = K_select==num_hmm;
            is_K_big(i) = K_select>num_hmm;
            is_K_less(i) = K_select<num_hmm;
            is_S_right(i) = sum(S_select==num_sta)/length(S_select);
            is_S_big(i) = sum(S_select>num_sta)/length(S_select);
            is_S_less(i) = sum(S_select<num_sta)/length(S_select);
            
            K_is(i) = K_select;
            S_is{i} = S_select;
            
            if cpt_dist
                
            intra_Dists = compute_hmms_dist(tmph3m.hmm,HMMs{i},Datas{i},tmph3m.groups,'vb');
            [inter_dists_vct,inter_Dists] = compute_inter_Centhmms_dist(tmph3m.hmm,Datas{i},tmph3m.groups,'vb');
            numerator_di = min(inter_dists_vct);
            denominator_di = max(cellfun(@max,intra_Dists));
            Dunn_index = numerator_di/ denominator_di; % high is better
            DI(i) = Dunn_index;
            
            end
            
        end
        
        
    case 'dic'
        
        if isfield(Group_hmms{1, 1},'vbhemopt')
            T = Group_hmms{1, 1}.vbhemopt.tau;
            isnew =1;
        else
            T = Group_hmms{1, 1}.vbhembopt.tau;
            isnew =0;
        end
        
        for i =1:iter
            vbh3m = Group_hmms{1, i};
            hmms = HMMs{i};
            P_d=[];
            DIC = [];
            for k =1:len_K
                for s = 1:len_S
                    if all(size(vbh3m.model_all)==[len_S,len_K])
                        vbh3mj = Group_hmms{i}.model_all{s, k};
                    
                    else
                    vbh3mj = vbh3m.model_all{1, k}.model_all_s{1, s};
                    end
                    [P_d(k,s),DIC(k,s)] = myDIC(hmms, vbh3mj,T,div_T);
                end
            end
            
            DICs{i} = DIC;
        end
        
        Output.DICs = DICs;
        Criteria = DICs;

        
    case 'vhbic'
        
        
        vh_hmms = Group_hmms;
        
        Nv = vh_hmms{1, 1}{1, 1}.hemopt.Nv;
        T = vh_hmms{1, 1}{1, 1}.hemopt.tau;
        N_bic = Nv*Kb*T;
        
        BICs = cell(iter,1);
        
        for it = 1:iter
            
            hmms_it = vh_hmms{it};
            BIC_it = zeros(len_K,len_S);
            for k = 1:len_K
                
                kk = K(k);
                for s = 1:len_S
                    
                    ss = S(s);
                    hmms_ks = hmms_it{k,s};
                    Num_pars = (kk-1) + kk*((ss-1) + ss*(ss-1) + ss*2*dim);
                    omega_ks = sum(hmms_ks.Z,1)./Kb;
                    log_ests = sum(sum(hmms_ks.Z.*( log(omega_ks).*ones(Kb,1) -log(hmms_ks.Z+1e-50) + Nv* hmms_ks.L_elbo1)));
                    
                    if div_T
                    BIC_it(k,s) = log(N_bic)*Num_pars-2*log_ests/T;

                    else
                    BIC_it(k,s) = log(N_bic)*Num_pars-2*log_ests;
                    end
                    
                end
            end
            
            BICs{it} = BIC_it;
        end
        
        Output.VHBICs = BICs;
        Criteria = BICs;
        
        
    case 'vhaic'
        
        
        vh_hmms = Group_hmms;
        
        Nv = vh_hmms{1, 1}{1, 1}.hemopt.Nv;
        T = vh_hmms{1, 1}{1, 1}.hemopt.tau;
        
        AICs = cell(iter,1);

        for it = 1:iter
            hmms_it = vh_hmms{(it)};
            AIC_it =[];
            for k = 1:len_K
                for s = 1:len_S
                    
                    hmms_ks = hmms_it{k,s};
                    Num_pars = K(k)*S(s)*(S(s)+2*dim)-1;
                    omega_ks = sum(hmms_ks.Z,1)./Kb;
                    log_ests = sum(sum(hmms_ks.Z.*( log(omega_ks).*ones(Kb,1) -log(hmms_ks.Z+1e-50) + Nv* hmms_ks.L_elbo1)));
                    if div_T
                    AIC_it(k,s) = 2*Num_pars-2*log_ests/T;
                    else
                    AIC_it(k,s) = 2*Num_pars-2*log_ests;
                    end
                end
            end
            
            AICs{it} = AIC_it;
        end
        
        
        Output.VHAICs = AICs;
        Criteria = AICs;
        
    case 'scaic'

        
        AICs = cell(iter,1);

        for it = 1:iter
            tmpdata = Datas{it};
            T = mean(cellfun(@(x) length(x{1,1}),tmpdata));
            
            AIC_it =[];
            for k = 1:len_K
                kk = K(k);
                for s = 1:len_S
                    ss = S(s);
                    Num_pars = (kk-1) + kk*((ss-1) + ss*(ss-1) + ss*2*dim);
                    if div_T
                        AIC_it(k,s) =-2*PPK_LL(k,s,it)/T + 2*Num_pars;
                    else
                        AIC_it(k,s) =-2*PPK_LL(k,s,it) + 2*Num_pars;
                    end
                end
            end
            AICs{it} = AIC_it;
        end
        
        
        Output.SCAIC = AICs;
        Criteria = AICs;
        
    case 'scbic'
        
        BICs = cell(iter,1);
        
        for it = 1:iter
            
            Adata2 = cat(1,Datas{it});
            Adata3 = cat(1,Adata2{:});
            N_bic  = sum(cellfun(@length,Adata3));
            tmpdata = Datas{it};
            T = mean(cellfun(@(x) length(x{1,1}),tmpdata));
            
            for k = 1:len_K
                kk = K(k);
                for s = 1:len_S
                    
                    ss = S(s);
                    Num_pars = (kk-1) + kk*((ss-1) + ss*(ss-1) + ss*2*dim);
                    if div_T
                        BIC_it(k,s) = -2*PPK_LL(k,s,it)/T + log(N_bic)*Num_pars;
                    else
                    BIC_it(k,s) = -2*PPK_LL(k,s,it) + log(N_bic)*Num_pars;
                    end
                end
            end
            
            BICs{it} = BIC_it;
        end
        
        Output.SCBIC = BICs;
        Criteria = BICs;
        
        
    case 'ccfd'
        
        AllCCFD_HMMs = Group_hmms;
        
        for i =1:iter
            
            CCFD_i = AllCCFD_HMMs{i};
            try
                ccfd_label = CCFD_i.label;
            catch
                continue
            end

            RI(i) =valid_RandIndex(true_label, ccfd_label);
            Puri(i) =Purity(true_label,ccfd_label);
            K_select = length(AllCCFD_HMMs{i}.ccfd_result.icl);
            
            is_K_right(i) = K_select==num_hmm;
            is_K_big(i) = K_select>num_hmm;
            is_K_less(i) = K_select<num_hmm;
            K_is(i) = K_select;
            
            if cpt_dist
                Centhmms = CCFD_i.hmms;
                tmpgruop = CCFD_i.groups;
                if length(Centhmms)==1
                    Dunn_index = 0; % high is better
                else
                    cent_ids= CCFD_i.ccfd_result.icl;
                    intra_Dists = compute_hmms_dist(Centhmms,HMMs{i},Datas{i},tmpgruop,'ccfd',cent_ids);
                    [inter_dists_vct,inter_Dists] = compute_inter_Centhmms_dist(Centhmms,Datas{i},tmpgruop,'ccfd',cent_ids);
                    numerator_di = min(inter_dists_vct);
                    denominator_di = max(cellfun(@max,intra_Dists));
                    Dunn_index = numerator_di/ denominator_di; % high is better
                end
                
                DI(i) = Dunn_index;

            end
            
        end
        
        case 'vhem'
        
        vh_hmms = Group_hmms;
        Nv = vh_hmms{1, 1}{1, 1}.hemopt.Nv;
        T = vh_hmms{1, 1}{1, 1}.hemopt.tau;
        
        for it = 1:iter
            
            hmms_it = vh_hmms{it};
            RI_vh = zeros(len_K,len_S);
            Purity0 = zeros(len_K,len_S);
            right_K_vh =zeros(len_K,len_S);
            big_K_vh =zeros(len_K,len_S);
            less_K_vh=zeros(len_K,len_S);
            right_S_vh =zeros(len_K,len_S);
            big_S_vh =zeros(len_K,len_S);
            less_S_vh=zeros(len_K,len_S);
            DI_vh =zeros(len_K,len_S);
            
            for k = 1:len_K
                for s = 1:len_S
                    
                    hmms_ks = hmms_it{k,s};
                    vh_label = hmms_ks.label;
                                        
                    RI_vh(k,s)=valid_RandIndex(true_label,vh_label);
                    [Purity0(k,s)] =Purity(true_label,vh_label);
                    
                    Nonempty_idx = hmms_ks.group_size~=0;
                    K_select= sum(Nonempty_idx);
                    
                    right_K_vh(k,s) = K_select==num_hmm;
                    big_K_vh (k,s) = K_select>num_hmm;
                    less_K_vh (k,s) = K_select<num_hmm;
                    
                    K_is_it(k,s) = K_select;
                    

                    if (len_S>1)
                        S_select = cellfun(@(x) sum(x.stats.emit_vcounts>1e-3),hmms_ks.hmms(Nonempty_idx));
                        %S_select = cellfun(@(x) length(x.prior),hmms_ks.hmms(Nonempty_idx));
                        
                        right_S_vh(k,s) = sum(S_select==num_sta)/length(S_select);
                        big_S_vh(k,s) = sum(S_select>num_sta)/length(S_select);
                        less_S_vh(k,s) = sum(S_select<num_sta)/length(S_select);
                        S_is_it{k,s} = S_select;
                        
                    end
                    
                    if cpt_dist
                        
                        Centhmms = hmms_ks.hmms(Nonempty_idx);
                        tmpgruop = hmms_ks.groups(Nonempty_idx);
                        
                        if length(Centhmms)==1
                            Dunn_index = 0; % high is better
                        else
                            intra_Dists = compute_hmms_dist(Centhmms,HMMs{it},Datas{it},tmpgruop,'vh');
                            [inter_dists_vct,inter_Dists] = compute_inter_Centhmms_dist(Centhmms,Datas{it},tmpgruop,'vh');
                            numerator_di = min(inter_dists_vct);
                            denominator_di = max(cellfun(@max,intra_Dists));
                            Dunn_index = numerator_di/ denominator_di; % high is better
                        end
                        DI_vh(k,s) = Dunn_index;
                    end
                    
                    

                    
                end
            end
            
            RI(it) = mean(RI_vh(:));
            Puri(it) = mean(Purity0(:));
            
            
            is_K_right(it) = mean(right_K_vh(:));
            is_K_big(it) = mean(big_K_vh(:));
            is_K_less(it) = mean(less_K_vh(:));
            
            if (len_S>1)
                is_S_right(it) = mean(right_S_vh(:));
                is_S_big(it) = mean(big_S_vh(:));
                is_S_less(it) = mean(less_S_vh(:));
                S_is{it} = S_is_it;
            end
            
            
            K_is{it} = K_is_it;  
            
            DI(it) = mean(DI_vh(:));
            
        end % end iter
        
end

%%%%% % fix the aic and bic with positive number
switch method
    case {'vhaic','vhbic'}
        tmp =cell2mat(cellfun(@(x) sum(sum(x<0)),Criteria,'UniformOutput',0));
        tmpind = find(tmp==1);
        for ik = 1:sum( tmp==1)
            ii = tmpind(ik);
            TMP_tmp = Criteria{ii};
            TMP_tmp(TMP_tmp<0)=+Inf;
            Criteria{ii}=TMP_tmp;
        end
end

%% selection

if (~strcmp(method,'vb'))&&(~strcmp(method,'ccfd'))&&(~strcmp(method,'vhem'))
    
    for i =1:iter
        
        % aic,bic,dic both the less, the better
        crite = Criteria{i};
        
        if size(crite,2) ~= len_S
            error('check');
        end
        
        if size(crite,1) ~= len_K
            error('check');
        end
        
        [min_indk,min_inds] = find(crite==min(crite(:)));
        
        switch method
            case {'vhaic','vhbic'}
                hmms_it = Group_hmms{i};
                hmms_ks = hmms_it{min_indk,min_inds};
                Nonempty_idx = (hmms_ks.group_size~=0);
                K_select= sum(Nonempty_idx); % VHEM itself has pruned the hmms and states
                %S_select = cellfun(@(x) length(x.prior),hmms_ks.hmms(hmms_ks.group_size~=0));
                S_select = cellfun(@(x) sum(x.stats.emit_vcounts>1e-3),hmms_ks.hmms(hmms_ks.group_size~=0));

                LABEL = hmms_ks.label;
                
                if cpt_dist
                    
                Centhmms = hmms_ks.hmms(Nonempty_idx);
                tmpgruop = hmms_ks.groups(Nonempty_idx);
                
                if length(Centhmms)==1
                    Dunn_index = 0; % high is better
                else
                    intra_Dists = compute_hmms_dist(Centhmms,HMMs{i},Datas{i},tmpgruop,'vh');
                    [inter_dists_vct,inter_Dists] = compute_inter_Centhmms_dist(Centhmms,Datas{i},tmpgruop,'vh');
                    
                end
                end
                
            case {'scaic','scbic'}
                
                hmms_it = Group_hmms{i};
                hmms_ks = hmms_it{min_indk,min_inds};
                K_select = length(hmms_ks.group_size); % VHEM itself has pruned the hmms and states
                S_select = cellfun(@(x) length(x.prior),hmms_ks.hmms);
                LABEL = hmms_ks.label;
                
                if cpt_dist
                Centhmms = hmms_ks.hmms;
                tmpgruop = hmms_ks.group;  % here is group, diff with vhem
                if length(Centhmms)==1
                    Dunn_index = 0; % high is better
                else
                    intra_Dists = compute_hmms_dist(Centhmms,HMMs{i},Datas{i},tmpgruop,'sc');
                    [inter_dists_vct,inter_Dists] = compute_inter_Centhmms_dist(Centhmms,Datas{i},tmpgruop,'sc');
                end
                end
                

                
            case {'dic'}
                
                vbh3m = Group_hmms{1, i};
                if all(size(vbh3m.model_all)==[len_S,len_K])
                    vbh3mj = vbh3m_remove_empty(vbh3m.model_all{min_inds, min_indk});
                else
                     vbh3mj = vbh3m_remove_empty(vbh3m.model_all{1, min_indk}.model_all_s{1, min_inds});
                end
%                 if isnew
%                   vbh3mj = vbh3m_remove_empty(vbh3m.model_all{1, min_indk}.model_all_s{1, min_inds});
%                 else
%                 vbh3mj = vbh3m_remove_empty(vbh3m.model_all{min_inds, min_indk});
%                 end
                
                K_select = length(vbh3mj.omega);
                S_select = cellfun(@(x) length(x.prior),vbh3mj.hmm);

                LABEL = vbh3mj.label;
                
                if cpt_dist
                Centhmms = vbh3mj.hmm;
                tmpgruop = vbh3mj.groups;
                
                if length(Centhmms)==1
                    Dunn_index = 0; % high is better
                else
                    intra_Dists = compute_hmms_dist(Centhmms,HMMs{i},Datas{i},tmpgruop,'dic');
                    [inter_dists_vct,inter_Dists] = compute_inter_Centhmms_dist(Centhmms,Datas{i},tmpgruop,'dic');
                    
                end
                end
                
        end
        
        RI(i) =valid_RandIndex(true_label, LABEL);
        Puri(i) =Purity(true_label, LABEL);
        
        is_K_right(i) = K_select==num_hmm;
        is_K_big(i) = K_select>num_hmm;
        is_K_less(i) = K_select<num_hmm;
        
        if (len_S>1)
            is_S_right(i) = sum(S_select==num_sta)/length(S_select);
            is_S_big(i) = sum(S_select>num_sta)/length(S_select);
            is_S_less(i) = sum(S_select<num_sta)/length(S_select);
            S_is{i} = S_select;
            
        end
        
        K_is(i) = K_select;
        
        if cpt_dist
        if length(Centhmms)>1
            numerator_di = min(inter_dists_vct);
            denominator_di = max(cellfun(@max,intra_Dists));
            Dunn_index = numerator_di/ denominator_di; % high is better
        end
        DI(i) = Dunn_index;
        end
        
        
    end
    
end

%% compute mean & std

mean_RI = mean(RI);
std_RI = std(RI);

if cpt_dist
    Output.DI = DI;
    
    mean_DI = mean(DI);
    std_DI = std(DI);
    
    Output.mean_DI = mean_DI;
    Output.std_DI = std_DI;    
end

mean_Purity = mean(Puri);
std_Purity = std(Puri);

mean_is_K_right = mean(is_K_right);
std_is_K_right = std(is_K_right);

mean_is_K_big = mean(is_K_big);
std_is_K_big = std(is_K_big);

mean_is_K_less = mean(is_K_less);
std_is_K_less = std(is_K_less);

if (len_S>1)&&(~strcmp(method,'ccfd'))
mean_is_S_right = mean(is_S_right);
std_is_S_right = std(is_S_right);

mean_is_S_big = mean(is_S_big);
std_is_S_big = std(is_S_big);

mean_is_S_less = mean(is_S_less);
std_is_S_less = std(is_S_less);
end


%% save the output
Output.RI = RI;
Output.Purity = Puri;

Output.mean_RI = mean_RI;
Output.std_RI = std_RI;

Output.mean_Purity = mean_Purity;
Output.std_Purity = std_Purity;

Output.mean_is_K_right = mean_is_K_right;
Output.std_is_K_right = std_is_K_right;

Output.mean_is_K_big = mean_is_K_big;
Output.std_is_K_big = std_is_K_big;

Output.mean_is_K_less = mean_is_K_less;
Output.std_is_K_less = std_is_K_less;

Output.K_is = K_is;

if (len_S>1)&&(~strcmp(method,'ccfd'))
    Output.mean_is_S_right = mean_is_S_right;
    Output.std_is_S_right = std_is_S_right;
    
    Output.mean_is_S_big = mean_is_S_big;
    Output.std_is_S_big = std_is_S_big;
    
    Output.mean_is_S_less = mean_is_S_less;
    Output.std_is_S_less = std_is_S_less;
    
    Output.S_is = S_is;
    
end

%Output.Dists = ADists;

