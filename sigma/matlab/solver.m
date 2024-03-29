function [pi_mat,d_star]=solver(params, algo_predictor)

% Will run an FP like loop. 
% Will need a function that computes the steady state given a
% policy, and the action distribution of that state.

[p_no,c_no] = size(params.l_user);

% Setup the initial policy
[vals,l_user_optim_idx] = min(params.l_user,[],2);
l_user_optim = zeros(c_no,p_no);
for idx = 1:p_no,
  l_user_optim(l_user_optim_idx(idx),idx)=1;
end
%l_user_optim
pi_vec = 0.7*reshape(l_user_optim,c_no*p_no,1)+0.3*ones(c_no*p_no,1)/c_no;
d_base = algo_predictor(reshape(pi_vec,c_no,p_no),params);
%pi_vec = ones(c_no*p_no,1)/c_no;

% Setup local optimisation parameters
A_robot = reshape(repmat(params.sigma_w,c_no,1).*params.l_robot',c_no*p_no,1);
A_user = reshape(repmat(params.sigma_w,c_no,1).*params.l_user',1,c_no*p_no);
pi_low = zeros(c_no*p_no,1);
pi_high = ones(c_no*p_no,1);
A_eq = [];
for idx = 1:p_no,
  A_eq = blkdiag(ones(1,c_no),A_eq);
end
b_eq = ones(p_no,1);

% We are improving the user's utility by this coefficient
threshold = 1.02;

% Loop through FP-like
err = 1.0;
counter = 1;
while (err > 1e-5),
  pi_mat = reshape(pi_vec,c_no,p_no);
  % Calculate steady state expert choice distribution
  [d,d_clear] = algo_predictor(pi_mat,params);
  %d(1:end-1)/sum(d(1:end-1))-d_clear
  % Calculate optimisation parameters for the next pi, and find it
  b = params.sigma_w*params.l_user*(d_clear-threshold*d(1:end-1))/...
      (threshold*d(end));
  
  f = @(x)(A_robot'*x);
%  [pi_vec_new,fval,exitflag] = ...
%      fmincon(f,pi_vec,A_user,b,A_eq,b_eq,pi_low,pi_high,[],...
%	      optimset('Display','off'));
  [pi_vec_new,fval,exitflag] = ...
      linprog(A_robot,A_user,b,A_eq,b_eq,pi_low,pi_high,pi_vec,...
	      optimset('Display','off'));
  % Measure difference between the old and the new pi_vector
  err = norm(pi_vec-pi_vec_new)+norm(d_base-d);
  if (mod(counter-1,10)==0)
    fprintf('At stage %d the error is %f\n',counter,err);
  end
  % Mix the new and old pi vectors
  counter = counter +1;
%  pi_vec = (1-1.0/counter)*pi_vec+(1.0/counter)*pi_vec_new;
%  d_base = (1-1.0/counter)*d_base+(1.0/counter)*d;
  alpha_loc = 0.05;
  pi_vec = (1-alpha_loc)*pi_vec+alpha_loc*pi_vec_new;
  d_base = (1-alpha_loc)*d_base+alpha_loc*d;
end
pi_mat = reshape(pi_vec,c_no,p_no);
d_star = algo_predictor(pi_mat,params)
