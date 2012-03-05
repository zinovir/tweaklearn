function [d,d_clear] = f_softmax_exp3_pred(pi_mat,params)

L = (params.sigma_w*params.l_user)';

L_star = params.sigma_w*diag(params.l_user*pi_mat);

d = exp(-params.eta*[L;L_star]);
d = d./sum(d);

d = (1-params.gamma)*d+params.gamma*ones(size(d))/length(d);

if (nargout == 2)
  d_clear = exp(-params.eta*L);
  d_clear = d_clear./sum(d_clear);
  d_clear = (1-params.gamma)*d_clear+...
	    params.gamma*ones(size(d_clear))/length(d_clear);
end
