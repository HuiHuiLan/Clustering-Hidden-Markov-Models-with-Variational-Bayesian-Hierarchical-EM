%% THE CODE FOR A NOVEL CLUSTER CENTER FAST DERMINATION CLUSTERING ALGORITHM
function [fitness,dc,icl,cl,rho,delta,halo] = CCFD(dist,percent,pur,k,dc)

% input distance matrix

if nargin <4
% determine a dc
k=3;
end

if nargin <5
% determine a dc
sda=sort(pur(:));
dc = sda(1) + (sda(end) - sda(1))*percent/100;
end


ND = size(dist,1);
% calculate rho_i and gamma_i for each data i
%%%%%%%%%%%%%%%% calculate rho %%%%%%%%%%%%%%%
for i=1:ND
  rho(i)=0.;
end

% % Gaussian kernel
% %
% for i=1:ND-1
%   for j=i+1:ND
%      rho(i)=rho(i)+exp(-(dist(i,j)/dc)*(dist(i,j)/dc));
%      rho(j)=rho(j)+exp(-(dist(i,j)/dc)*(dist(i,j)/dc));
%   end
% end

% "Cut off" kernel

for i=1:ND-1
 for j=i+1:ND
   if (dist(i,j)<dc)
      rho(i)=rho(i)+1.;
      rho(j)=rho(j)+1.;
   end
 end
end
%%%%%%%%%%%%%%%% calculate delta %%%%%%%%%%%%%%%

maxd=max(max(dist));

[rho_sorted,ordrho]=sort(rho,'descend');
delta(ordrho(1))=-1.;
nneigh(ordrho(1))=0;  % the nearest neighbr

for ii=2:ND
   delta(ordrho(ii))=maxd;
   for jj=1:ii-1
     if(dist(ordrho(ii),ordrho(jj))<delta(ordrho(ii)))
        delta(ordrho(ii))=dist(ordrho(ii),ordrho(jj));
        nneigh(ordrho(ii))=ordrho(jj);
     end
   end
end
delta(ordrho(1))=max(delta(:));

%%% save the decision graph

% 
% tt=plot(rho(:),delta(:),'o','MarkerSize',5,'MarkerFaceColor','k','MarkerEdgeColor','k');
% title ('Decision Graph','FontSize',15.0)
% xlabel ('\rho')
% ylabel ('\delta')

%%%%%%%%%%%%%%%% gamma %%%%%%%%%%%%%%%

for i=1:ND
  %ind(i)=i;
  gamma(i)=rho(i)*delta(i);
end

mgamma = mean(gamma);
% 
% figoptsx = {'Position', [100 100 350 300]};
% figure(figoptsx{:})
% [HN, HX] = hist(gamma(:),100);
% bar(HX, HN)
% xlabel('\gamma');
% ylabel('count');
% grid on
% hold on

%covgamma= cov(gamma);

% drop some data points
n = 2;
threshold = n*mgamma;
for i =1:ND
    if (gamma(i)>threshold)
        useid(i) =0;
    else
        useid(i) = 1;
    end
end

mgamma = mean(gamma(1,useid(:)==1));
covgamma= cov(gamma(1,useid(:)==1));
 
% plot the normal distribution

% figoptsx = {'Position', [100 100 350 300]};
% figure(figoptsx{:})

% x=linspace(mgamma-2*sqrt(covgamma),mgamma+2*sqrt(covgamma),201)';
% y1=normpdf(x,mgamma,sqrt(covgamma));
% plot(x,y1)     


% select the singular points
nfivesigma = mgamma+5*sqrt(covgamma);
singid1 = find(gamma>nfivesigma);

pfivesigma = mgamma-5*sqrt(covgamma);
singid2 = find(gamma<pfivesigma);

singid = [singid1,singid2];

  
if length(singid)<1
    error('NO SINGULAR POINTS')
end

% rho_star = (rho-min(rho))/(max(rho)-min(rho));
% delta_star = (delta-min(delta))/(max(delta)-min(delta));

% plot(rho_star,delta_star,'o')
% grid on
% hold on
% plot([0,1],[0,3])
% plot([0,1],[0,1/3])

%%%%%%%%%%%%%%%%
k1=k;
k_star1 = (k1*(max(delta)-min(delta)) +min(delta) )/((max(rho)-min(rho))+min(rho));

k2 = 1/k;
k_star2 = (k2*(max(delta)-min(delta)) +min(delta) )/((max(rho)-min(rho))+min(rho));
% 
% figure
% plot(rho,delta,'o')
% grid on
% hold on
% plot([0,0.07/k_star1],[0,0.07])
% plot([0,18],[0,k_star2*18])



NCLUST=0;
for i=1:ND
  cl(i)=-1;
end

for i=1:length(singid)
    
  j = singid(i);
  if ((rho(j)/delta(j)<(1/k_star2))&&(delta(j)/rho(j)<k_star1))
      
     NCLUST=NCLUST+1;
     cl(j)=NCLUST;
     icl(NCLUST)=j;
  end
end
fprintf('NUMBER OF CLUSTERS: %i \n', NCLUST);

if NCLUST<1
    error('NO SINGULAR POINTS')
end
%disp('Performing assignation')

%assignation
for i=1:ND
  if (cl(ordrho(i))==-1)
    cl(ordrho(i))=cl(nneigh(ordrho(i)));
  end
end


%halo
for i=1:ND
  halo(i)=cl(i);
end


if (NCLUST>1)
  for i=1:NCLUST
    bord_rho(i)=0.;
  end
  for i=1:ND-1
    for j=i+1:ND
      if ((cl(i)~=cl(j))&& (dist(i,j)<=dc))
        rho_aver=(rho(i)+rho(j))/2.;
        if (rho_aver>bord_rho(cl(i))) 
          bord_rho(cl(i))=rho_aver;
        end
        if (rho_aver>bord_rho(cl(j))) 
          bord_rho(cl(j))=rho_aver;
        end
      end
    end
  end
  for i=1:ND
    if (rho(i)<bord_rho(cl(i)))
      halo(i)=0;
    end
  end
end



for i=1:NCLUST
  nc=0;
  nh=0;
  for j=1:ND
    if (cl(j)==i) 
      nc=nc+1;
    end
    if (halo(j)==i) 
      nh=nh+1;
    end
  end
  fprintf('CLUSTER: %i CENTER: %i ELEMENTS: %i CORE: %i HALO: %i \n', i,icl(i),nc,nh,nc-nh);
end


%% compute fitness
fit1 = 0;
for j =1:NCLUST   
    jiterm = cl==j;
    tmpdist = dist(:,icl(j));
    fit1 = fit1 + sum(tmpdist(jiterm))/length(jiterm);
end
fit1 = fit1/NCLUST;

if NCLUST>1
tmpfit2 = zeros(NCLUST,NCLUST);
for j =1:NCLUST  
    cen1 = icl(j);
    for i = (j+1):NCLUST
        cen2 = icl(i);
        
        tmpfit2(j,i) = dist(cen1,cen2);
        tmpfit2(i,j) = dist(cen1,cen2);

    end
end
fit2 = sum(tmpfit2(:))/NCLUST/(NCLUST-1);
else
    fit2=0;
end
fitness = fit2/fit1;

    
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
