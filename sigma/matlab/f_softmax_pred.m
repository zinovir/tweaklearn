function [d] = f_softmax_pred(pi_mat,params)

L = (params.sigma_w*params.l_user)';

L_star = params.sigma_w*diag(params.l_user*pi_mat);

d = exp(-params.eta*[L;L_star]);
d = d./sum(d);

d = (1-params.gamma)*d+params.gamma*ones(size(d))/length(d);
