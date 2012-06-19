function []=runner(conf_file_idxs)

conf_file_base = 'config_params_v';
pi_name_base = 'policy_';

pi_alg_names = {'egreedy','softmax','softmax_exp3'};
pi_alg_funs = {@f_egreedy_pred,@f_softmax_pred,@f_softmax_exp3_pred};

for idx = conf_file_idxs,
  fprintf('Configuration No %d \n',idx);
  % Recreating configuration file name
  conf_file = strcat(conf_file_base,int2str(idx),'.mat');
  % Recovering configuration data
  src_params = get_src_params(conf_file);
  
  % Setting the configuration invariant portions of solution policy
  policy.sigma = src_params.sigma;
  policy.sigma_p = src_params.sigma_p;

  % Running over all possible user algorithms
  for idx_alg = 1:length(pi_alg_names),
    fprintf('Running algorithm %s\n',pi_alg_names{idx_alg});
    % Recreating solution file name
    pi_file = strcat(pi_name_base,pi_alg_names{idx_alg},...
		     '_v',int2str(idx),'.mat');
    % Calculating solution with a given algorithm
    [policy.pi,policy.d_star] = solver(src_params,pi_alg_funs{idx_alg});
    % Saving the outcome
    save(pi_file,'-struct','policy');
  end
end
