function plot_prior(prior)
% plot_prior - plot fixations on an image
%
%  plot_prior(prior)
%
% INPUT: 
%   prior - prior distribution
%
% ---
% Eye-Movement analysis with HMMs (emhmm-toolbox)
% Copyright (c) 2017-01-13
% Antoni B. Chan, Janet H. Hsiao, Tim Chuk
% City University of Hong Kong, University of Hong Kong

% 2017-05-24: ABC - updated to use new color_list
% 2019-02-20: v0.75 - text changes with figure size



K = length(prior);

nfontsize2 = 0.25/3;

color = get_color_list(K);

set(gca, 'XTick', [1:K]);
hold on
for i=1:K
  bar(i, prior(i), 'FaceColor', color{i});
end
grid on
title('prior');
ylabel('probability');
xlabel('ROI');
set(gca, 'FontUnits', 'normalized', 'FontSize', nfontsize2);
