function [x0 problem] = make_matlab_solve();
% Set up the tweaklearn problem in Matlab and solve it

% For now, problem is 4 x 4
wsize = 4;
problem.NUM_STATES = wsize * wsize;

% This problem has 5 actions {remain, north, south, east, west}
problem.NUM_ACTS = 5;

% Only allow 4 steps of policy iteration
problem.MAX_TIME = 4;
% Set the temperature / learning rate for policy iteration
problem.TAU = [0.99, .95, .9, .8];

% Target state is '15' (left of the bottom right corner)
% Reward function is -1 for all states except for this one
target_state = 15;

% Set discount factor
problem.GAMMA = 0.95;

% Target policy (state, action)
problem.PISTAR = [
0 0 1 0 0;
0 0 0 1 0;
0 0 0 1 0;
0 0 1 0 0;
0 0 1 0 0;
0 0 0 0 1;
0 0 0 1 0;
0 0 1 0 0;
0 0 1 0 0;
0 0 1 0 0;
0 0 1 0 0;
0 0 1 0 0;
0 0 0 1 0;
0 0 0 1 0;
1 0 0 0 0;
0 0 0 0 1;
];

% Add some minor noise to the policy to remove zeros
problem.PISTAR = (1 - 0.001 * problem.NUM_ACTS) * problem.PISTAR + 0.001;

% Initial policy of the learning algorithm (pi_0)
% (Play first action at each state)
% TODO: change this first policy?
problem.PIZERO = sparse(problem.NUM_ACTS, problem.NUM_STATES);
problem.PIZERO(1,:) = 1;

% Get 'passive dynamics' of system
[P problem.A] = genmdp(wsize);

TP(:, 1, :) = P.x;
TP(:, 2, :) = P.n;
TP(:, 3, :) = P.s;
TP(:, 4, :) = P.e;
TP(:, 5, :) = P.w;

problem.TSTAR = TP;

problem.C = -ones(problem.NUM_STATES, problem.NUM_ACTS, problem.NUM_STATES);
% Target state has a reward value of +1
problem.C(target_state, :, :) = 1;

% Constraints
totalsize = problem.MAX_TIME * problem.NUM_STATES * 2 + problem.MAX_TIME * problem.NUM_STATES * problem.NUM_ACTS * problem.NUM_STATES;
% Lower and upper bounds (currently only bounds on T_{u_t} \in [0,1])
lb = -Inf(totalsize,1);
ub = Inf(totalsize,1);
i_a = problem.MAX_TIME * problem.NUM_STATES + 1;
i_b = i_a + problem.MAX_TIME * problem.NUM_STATES * problem.NUM_ACTS * problem.NUM_STATES  - 1;
lb(i_a:i_b) = 0;
ub(i_a:i_b) = 1;

% No constraints of this form
Aeq = [];
beq = [];

% Constraints to ensure probabilities sum to 1 in T_{u_t}

A = sparse(problem.MAX_TIME * problem.NUM_ACTS * problem.NUM_STATES, totalsize);

i_c = 1;
tut = zeros(problem.MAX_TIME, problem.NUM_STATES, problem.NUM_ACTS, problem.NUM_STATES); 
for t = 1:problem.MAX_TIME
  for a = 1:problem.NUM_ACTS
    for s = 1:problem.NUM_STATES
      tut(t, :, a, s) = 1;
      
      A(i_c, i_a:i_b) = tut(:)';
      i_c = i_c + 1;
      tut(t, :, a, s) = 0;
    end
  end
end

b = ones(problem.MAX_TIME * problem.NUM_ACTS * problem.NUM_STATES, 1);

% Starting point (No modifications to passive dynamics at any step)

% Chose starting point that satisfies the constraints
% TODO: Q0 and V0 are possibly wrong, but see if it affects fmincon
Q0 = zeros(problem.NUM_STATES,1); % Satisfies (trivially) constraints on Q.

% Calculate starting V by iterating through the formula a few times
V0 = zeros(problem.NUM_STATES,1); % Satisfies (trivially) constraints on Q.
for i = 1:100
  Vold= V0;
  for s = 1:problem.NUM_STATES
    V0(s) = 0;
    for a = 1:problem.NUM_ACTS
      for sp = 1:problem.NUM_STATES
        V0(s) += TP(s, a, sp) * problem.PIZERO(a, s) * (problem.C(sp, a, s) + problem.GAMMA*Vold(sp));
      end
    end
  end
end

x0 = [repmat(Q0(:), problem.MAX_TIME, 1); 
      repmat(TP(:), problem.MAX_TIME, 1); 
      repmat(V0(:), problem.MAX_TIME, 1);]; 

options = optimset('Display','iter','Algorithm','active-set','GradConstr','on');

% Run solver
[x fval] = fmincon(@(x)(tweaklearn_func(x, problem)), x0, A, b, Aeq, beq, lb, ub, @(x)(tweaklearn_nonlcon(x, problem)), options);

% Unpack values
