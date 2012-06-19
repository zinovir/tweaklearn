list_spa_names = {'worst','selfish','best'};

exp_no = 20;
win_size = 100;
bins = [0:4];


data = load('tmp0.mat');
%max_idx = length(data.choices);

[max_idx,outcome_len] = size(data.outcome);
mat_stat = zeros(length(bins),max_idx-win_size,exp_no);
e_costs_full = zeros(max_idx,exp_no);
e_costs_smoothed = zeros(max_idx-win_size,exp_no);

for exp_idx = 1:exp_no,
  data = load(strcat('tmp',int2str(exp_idx-1),'.mat'));
  for idx = 1:max_idx,
    e_costs_full(idx,exp_idx)=...
	data.outcome(idx,outcome_len/2+1+...
		     data.polls(idx,data.choices(idx)+1));
  end
  for idx = 1:max_idx-win_size,
    mat_stat(:,idx,exp_idx)=hist(data.choices(idx:idx+win_size),bins);
    e_costs_smoothed(idx,exp_idx)=mean(e_costs_full(idx:idx+win_size,exp_idx));
  end
end

labels = {'E_0','E_1','E_2','E_3','E_4'};
styles = {'g-.','b-.','r-.','m-.','k-'};

% for exp_idx = 1:exp_no,
%   figure;
%   for line_idx = 1:length(bins),
%     plot([1:max_idx-win_size],mat_stat(line_idx,:,exp_idx),styles{line_idx});
%     hold on;
%   end
%   legend(labels)
% end

figure
ms = mean(mat_stat,3);
sz = size(ms);
ms = ms./repmat(sum(ms,1),sz(1),1);
for line_idx = 1:length(bins),
  plot([1:max_idx-win_size],ms(line_idx,:),styles{line_idx});
  hold on
end
legend(labels);
title('Frequencies of expert use');

% Plotting  SPA's costs
figure
plot([1:max_idx-win_size],mean(e_costs_smoothed,2));
title('Smoothed costs');
figure
e_mat = cumsum(e_costs_full,1);
plot([1:max_idx],mean(e_mat,2));
title('True costs');
