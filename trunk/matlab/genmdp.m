function [P A] = genmdp(wsize);
% Generates the underlying MDP for the problem by defining the actions and state transition probabilities.
% Grid world of size wsize x wsize
% P - Set of state transition matrices for the 5 actions
% P.n, P.s, P.w, P.e, P.x - Move north, south, west or east, or stay still (P.x)

% Generate land
s = 1;

A = (wsize)*randn(s);
do
  s = s*2;
  B = (wsize/s)*randn(s);
  for x = 1:s
    for y = 1:s
      B(x,y) = B(x,y) + A(ceil(x/2), ceil(y/2));
    end
  end

  A = B;

until s >= wsize

%diffuse
B = A;
for d=1:2
  for x = 1:wsize
    for y = 1:wsize
      if (x == 1 || x == wsize) && (y == 1 || y == wsize)
        %corner
      elseif (x == 1 || x == wsize)
        B(x,y) = mean(mean(A(x,y-1:y+1)));
      elseif (y == 1 || y == wsize)
        B(x,y) = mean(mean(A(x-1:x+1,y)));
      else
        B(x,y) = mean(mean(A(x-1:x+1,y-1:y+1)));
      end
    end
  end
  
  A = B;
end

% Make 10 different altitude levels, 0-9
A = A - min(min(A));
A = 10*A / max(max(A));
A = round(A);

% Calculate transition probabilities for state transitions

% Staying still always succeeds
P.x = eye(wsize^2); 

% North, south, east, west
P.n = eye(wsize^2); 
P.s = eye(wsize^2); 
P.e = eye(wsize^2); 
P.w = eye(wsize^2); 


% At the moment, probabilities don't reflect gradient of slope, just sign of gradient
% TODO: agent should have probability of changing direction, at present this is not the case
% Base Probability of struggling uphill
Bstrug = 0.9;
% Base Probability of stilding downhill
Bslide = 0.2;

for x = 1:wsize
  for y = 1:wsize
    %current state
    s1 = (y - 1)*wsize + x;

    % Check north
    if(y < wsize)
      s2 = y*wsize + x;
      s3 = (y + 1)*wsize + x;
      
      % Slope
      diff = A(x,y+1) - A(x,y);
      % Three probabilities.  Stay (struggle), move forward, slip forward 2 squares.  Should sum to 1
      
      if (y + 1) < wsize
        slip = Bslide / (1+exp(2 + 2*diff));
      else
        slip = 0;
      end
      stay = Bstrug / (1+exp(2 - 2*diff));
      move = 1 - slip - stay;

      P.n(s1, s1) = stay;
      P.n(s1, s2) = move;
      if (y + 1) < wsize
        P.n(s1, s3) = slip;
      end
    end

    % Check south
    if(y > 1)
      s2 = (y - 2)*wsize + x;
      s3 = (y - 3)*wsize + x;
      
      % Slope
      diff = A(x,y-1) - A(x,y);
      % Three probabilities.  Stay (struggle), move forward, slip forward 2 squares.  Should sum to 1
      
      if y - 1 > 1
        slip = Bslide / (1+exp(2 + 2*diff));
      else
        slip = 0;
      end
      stay = Bstrug / (1+exp(2 - 2*diff));
      move = 1 - slip - stay;

      P.s(s1, s1) = stay;
      P.s(s1, s2) = move;
      if y - 1 > 1
        P.s(s1, s3) = slip;
      end
    end

    % Check west
    if(x > 1)
      s2 = (y - 1)*wsize + x - 1;
      s3 = (y - 1)*wsize + x - 2;
      
      % Slope
      diff = A(x-1,y) - A(x,y);
      % Three probabilities.  Stay (struggle), move forward, slip forward 2 squares.  Should sum to 1
      
      if x - 1 > 1
        slip = Bslide / (1+exp(2 + 2*diff));
      else
        slip = 0;
      end
      stay = Bstrug / (1+exp(2 - 2*diff));
      move = 1 - slip - stay;

      P.w(s1, s1) = stay;
      P.w(s1, s2) = move;
      if x - 1 > 1
        P.w(s1, s3) = slip;
      end
    end

    % Check east
    if(x < wsize)
      s2 = (y - 1)*wsize + x + 1;
      s3 = (y - 1)*wsize + x + 2;
      
      % Slope
      diff = A(x+1,y) - A(x,y);
      % Three probabilities.  Stay (struggle), move forward, slip forward 2 squares.  Should sum to 1
      
      if (x + 1) < wsize
        slip = Bslide / (1+exp(2 + 2*diff));
      else
        slip = 0;
      end
      stay = Bstrug / (1+exp(2 - 2*diff));
      move = 1 - slip - stay;

      P.e(s1, s1) = stay;
      P.e(s1, s2) = move;
      if (x + 1) < wsize
        P.e(s1, s3) = slip;
      end
    end
    
  end
end

% Add some very small uniform probability to all transitions (i.e. nothing is impossible)
small = 0.000001;
P.n = (P.n + small) / (1 + wsize*small); 
P.s = (P.s + small) / (1 + wsize*small); 
P.w = (P.w + small) / (1 + wsize*small); 
P.e = (P.e + small) / (1 + wsize*small); 
P.x = (P.x + small) / (1 + wsize*small); 

