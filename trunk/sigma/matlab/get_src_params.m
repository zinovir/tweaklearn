function [src_params] = get_src_params(file_name)

% This function is designed to collect together all sources info

% % Number of sources
% src_params.n_choice = 4;

% % Number of experts to be running over it
% src_params.n_experts = 5;

% % Means of sources (including the robots costs, grouped first for
% % user, then the robot)
% src_params.mu = [3.0,2.0,1.5,4.0];

% % Covariance of sources
% src_params.sigma = diag([0.1,0.7,0.1,0.1]);

% % Discrepancy cut_off
% src_params.discr = 4.1;

% src_params.beta=0.01;
% src_params.gamma=0.1;
% src_params.beta_spec=0.05;
% src_params.eta=15.0;
% src_params.epsilon=0.05;

% First use data supplied by the Python code
src_params = load(file_name);


% Calculate sigma points
[n_x,n_x]=size(src_params.sigma);
s_tmp = sqrtm(n_x*src_params.sigma);
src_params.sigma_p = repmat(src_params.mu,1,2*n_x)+[s_tmp,-s_tmp];

src_params.sigma_w = ones(1,2*n_x)/(2.0*n_x);

% Calculate the utilities -- take into account the linear mapping
mu_user = src_params.mu(1:src_params.n_choice);
sigma_user = sqrt(diag(src_params.sigma(1:src_params.n_choice,...
					1:src_params.n_choice)));
x_min = min(mu_user-src_params.discr*sigma_user);
x_max = max(mu_user+src_params.discr*sigma_user);

src_params.a_coeff = 1.0/(x_max-x_min);
src_params.b_coeff = -x_min*src_params.a_coeff;

src_params.l_user = src_params.a_coeff*...
    src_params.sigma_p(1:src_params.n_choice,:)'+src_params.b_coeff;
src_params.l_robot = src_params.sigma_p(src_params.n_choice+1:end,:)';


