function CCFD_plot(rho,delta,k,NCLUST,icl)

colormap=linspecer((NCLUST+2));

fsz=20;
lw=6;
msz=20;


tt=plot(rho(:),delta(:),'o','MarkerSize',0.5*msz,'MarkerFaceColor','k','MarkerEdgeColor','k');
% title ('Decision Graph','FontSize',15.0)
% xlabel ('\rho')
% ylabel ('\delta')

%max_rho = ceil(max(rho));
%mocap
%max_delta = 800;
% music
% max_delta = 70;
% max_rho = 120;

%%%%% hand
% max_delta = 4;
% max_rho = 25;

% syn
max_delta = 0.08;
max_rho = 18;
% plot k
k1=k;
k_star1 = (k1*(max(delta)-min(delta)) +min(delta) )/((max(rho)-min(rho))+min(rho));

k2 = 1/k;
k_star2 = (k2*(max(delta)-min(delta)) +min(delta) )/((max(rho)-min(rho))+min(rho));
% 

hold on

plot([0,max_delta/k_star1],[0,max_delta],'-','LineWidth',lw,'color',colormap(3,:));
plot([0,max_rho],[0,k_star2*max_rho],'-','LineWidth',lw,'color',colormap(3,:));

%yticks([0, 0.02, 0.04, 0.06, 0.08]);

cmap=colormap;
for i=1:NCLUST
   %ic=int8((i*64.)/(NCLUST*1.));
   clr = i;
   hold on
   plot(rho(icl(i)),delta(icl(i)),'o','MarkerSize',msz,'MarkerFaceColor',cmap(clr,:),'MarkerEdgeColor',cmap(clr,:));
end
%hold on
%plot(rho(icl(1)),delta(icl(1)),'o','MarkerSize',msz,'MarkerFaceColor',cmap(2,:),'MarkerEdgeColor',cmap(2,:));
% hold on
% plot(rho(icl(2)),delta(icl(2)),'o','MarkerSize',msz,'MarkerFaceColor',cmap(4,:),'MarkerEdgeColor',cmap(4,:));


% gcf: Figure 
% gca: axes 

%set(gca,'FontSize',fsz);

set(gcf,'Position',[400 100 550 400]);
set(gcf,'Renderer','Painters');

title(['Decision Graph'])
xlabel('\rho');
ylabel('\delta');
set(gca,'linewidth',1,'fontsize',20,'fontname','Times');


