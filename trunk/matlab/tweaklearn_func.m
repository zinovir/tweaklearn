function [f g] = tweaklearn_func(x, problem);
% The cost function for the tweaklearn problem


% Reshape x into a more manageable form

% Extract Q(TIME, STATE), Teeyou(TIME, STATE, ACTION, STATE), Vee(TIME, STATE)

a = 1; b = a + problem.MAX_TIME * problem.NUM_STATES - 1;
Q = reshape(x(a:b), problem.MAX_TIME, problem.NUM_STATES);

a = b + 1; b = a + problem.MAX_TIME * problem.NUM_STATES * problem.NUM_ACTS * problem.NUM_STATES  - 1;
Teeyou = reshape(x(a:b), problem.MAX_TIME, problem.NUM_STATES, problem.NUM_ACTS, problem.NUM_STATES);

a = b + 1; b = a + problem.MAX_TIME * problem.NUM_STATES - 1;
Vee = reshape(x(a:b), problem.MAX_TIME, problem.NUM_STATES);


f = 0;

% Partial derivatives
pdQ = zeros(size(Q));
pdTeeyou = zeros(size(Teeyou));
pdVee = zeros(size(Vee)); % partial derivative wrt Vt(s) is 0

for t = 1:problem.MAX_TIME

  % pi_t(a|s)
  Pitee = zeros(problem.NUM_ACTS, problem.NUM_STATES);
  for s = 1:problem.NUM_STATES
    
    % Calculate pi_t(a|s) for each action
    for a = 1:problem.NUM_ACTS
      Pitee(a, s) = 0;
      
      for sp = 1:problem.NUM_STATES
        Pitee(a, s) = Pitee(a, s) + Teeyou(t, sp, a, s) * (problem.C(sp,a,s) + problem.GAMMA * Vee(t,sp));
      end
      Pitee(a, s) = exp(problem.TAU(t) * Pitee(a, s));
    end

    % Sum is Z_t(s)
    Pitee(:, s) = Pitee(:, s) ./ sum(Pitee(:, s));
  end
  
  for s = 1:problem.NUM_STATES
    for a = 1:problem.NUM_ACTS
      
      % Calculate D_t^KL(s, a)
      dtkl = 0;
      for sp = 1:problem.NUM_STATES          
        for ap = 1:problem.NUM_ACTS    
          if Teeyou(t, sp, a, s) * Pitee(ap, sp) != 0
          	dtkl = dtkl + Teeyou(t, sp, a, s) * Pitee(ap, sp) * log((Teeyou(t, sp, a, s) * Pitee(ap, sp))/(problem.TSTAR(s, a, sp)*problem.PISTAR(sp,a)));          	
          end
          
          if Pitee(ap, sp) != 0
            pdTeeyou(t, sp, a, s) = pdTeeyou(t, sp, a, s) + Pitee(ap, sp) * (log((Teeyou(t, sp, a, s) * Pitee(ap, sp))/(problem.TSTAR(s, a, sp)*problem.PISTAR(sp,a))) + 1);
          end
        end
        
        pdTeeyou(t, sp, a, s) = pdTeeyou(t, sp, a, s) * Pitee(a, s) * Q(t, s);
      end
      
      % Objective function
      f = f + Pitee(a, s) * Q(t,s) * dtkl;
      
      % Partial derivative of Q
      pdQ(t, s) = pdQ(t,s) + Pitee(a,s) * dtkl;
    end
  end
end


if nargout > 1
   g = [pdQ(:); pdTeeyou(:); pdVee(:)];
end

