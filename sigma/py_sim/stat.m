exp_no = 75;
data = load('tmp0.mat');
max_idx = length(data.choices);
win_size = 100;
bins = [0:4];

mat_stat = zeros(length(bins),max_idx-win_size,exp_no);

for exp_idx = 1:exp_no,
  data = load(strcat('tmp',int2str(exp_idx-1),'.mat'));
  for idx = 1:max_idx-win_size,
    mat_stat(:,idx,exp_idx)=hist(data.choices(idx:idx+win_size),bins);
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
legend(labels)
