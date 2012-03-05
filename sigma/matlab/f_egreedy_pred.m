function [d,d_clear] = f_egreedy_pred(pi_mat,params)

L = (params.sigma_w*params.l_user)';

L_star = params.sigma_w*diag(params.l_user*pi_mat);

x = [L;L_star];

[vals, x_idx] = min(x);

d = zeros(size(x));
d(x_idx)=1;
d = d./sum(d);

d = (1-params.epsilon)*d+params.epsilon*ones(size(d))/length(d);

if (nargout == 2)
  [vals,l_idx] = min(L);
  d_clear = zeros(size(L));
  d_clear(l_idx) = 1;
  d_clear = d_clear./sum(d_clear);
  d_clear = (1-params.epsilon)*d_clear+params.epsilon*ones(size(L))/length(L);
end

