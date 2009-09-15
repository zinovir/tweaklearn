function [c, ceq, gradc, gradceq] = tweaklearn_nonlcon(x, problem);
% Non-linear constraints for the tweaklearn problem

% Reshape x into a more manageable form

% Extract Q(TIME, STATE), Teeyou(TIME, STATE, ACTION, STATE), Vee(TIME, STATE)

a = 1; b = a + problem.MAX_TIME * problem.NUM_STATES - 1;
Q = reshape(x(a:b), problem.MAX_TIME, problem.NUM_STATES);

a = b + 1; b = a + problem.MAX_TIME * problem.NUM_STATES * problem.NUM_ACTS * problem.NUM_STATES  - 1;
Teeyou = reshape(x(a:b), problem.MAX_TIME, problem.NUM_STATES, problem.NUM_ACTS, problem.NUM_STATES);

a = b + 1; b = a + problem.MAX_TIME * problem.NUM_STATES - 1;
Vee = reshape(x(a:b), problem.MAX_TIME, problem.NUM_STATES);


% No inequality constraints
c = [];
gradc = [];


% Only need to encode G1 and G4 from the notes (G2 and G3 are incorporated into cost function)
% c holds G1 rows followed by G4 rows (t * s of each)
ceq = zeros(2 * problem.MAX_TIME * problem.NUM_STATES,1);

% Gradient (one column per constraint)
gradceq = sparse(prod(size(x)), size(c, 1));


Pitee = problem.PIZERO;

for t = 1:problem.MAX_TIME
  
  % pi_t(a|s)
  
  % G1 uses pi_{t-1}
  Piteeold = Pitee;
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
    g4gradQ = 0;
    g1gradT = 0;
    g4gradT = 0;
    g1gradV = 0;
    
    indexG1 = t + (s-1)*problem.MAX_TIME;
    indexG4 = indexG1 + problem.MAX_TIME * problem.NUM_STATES; 
    
    for a = 1:problem.NUM_ACTS
      for sp = 1:problem.NUM_STATES
        % G1
        ceq(indexG1) +=  Teeyou(t, sp, a, s) * Piteeold(a, s) * (problem.C(sp, a, s) + problem.GAMMA * Vee(t, sp));
        
        g1gradT = g1gradT + Piteeold(a, s) * (problem.C(sp, a, s) + problem.GAMMA * Vee(t, sp));
        
        % G4
        ceq(indexG4) += Teeyou(t, s, a, sp) * Pitee(a, sp) * Q(t, sp);
        
        g4gradT = g4gradT + Pitee(a, sp) * Q(t, sp);
      end
      
      g4gradQ = g4gradQ + Teeyou(t, s, a, s) * Pitee(a, s) - 1;
      
      g1gradV = g1gradV + Teeyou(t, s, a, s) * Piteeold(a, s) * problem.GAMMA - 1;
      
    end
    
    % G1
    ceq(indexG1) += -Vee(t, s);
    
    % G4
    ceq(indexG4) +=- Q(t, s);
    
    % Corresponding gradients
    
    % Q
    a = 1;
    
    % gradient for G1 wrt Q is 0
    % gradient for G4 wrt Q
    gradceq((a + indexG1 - 1), indexG4) = g4gradQ;
    
 
 	% T_{u_t}
    a = a + problem.MAX_TIME * problem.NUM_STATES;
    
    % gradient for G1 wrt T_u_t
    gradceq((a + indexG1 - 1), indexG1) = g1gradT;
    
    % gradient for G4 wrt T_u_t
    gradceq((a + indexG1 - 1), indexG4) = g4gradT;


	% V
	a = a + problem.MAX_TIME * problem.NUM_STATES * problem.NUM_ACTS * problem.NUM_STATES;

    % gradient for G1(V_t,s) wrt V_t(s)
    gradceq((a + indexG1 - 1), indexG1) = g1gradV;

	% gradient for G4 wrt V is 0    
    
  end
  
end
