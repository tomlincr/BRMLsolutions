function ex23_3
% Modified from DEMOHMMINFERENCESIMPLE
import brml.* 

H = 3; % number of Hidden states
V = 2; % number of Visible states
T = 3; % length of the time-series

% setup the HMM
phghm = [0.5 0 0; 0.3 0.6 0; 0.2 0.4 1];% transition matrix p(h(t)|h(t-1))
pvgh = [0.7 0.4 0.8; 0.3 0.6 0.2];% emission matrix p(v(t)|h(t))
ph1 = [0.9; 0.1; 0.0]; % initial p(h) NB must be as col, hence ; separators
v = [1 2 1]; % initial observations vector

disp('4. Filtering: p(h(3)|v(1:3))')
[alpha,loglik]=HMMforward(v,phghm,ph1,pvgh); % Returns: normalised alphas, log likelihood
alphaH3 = alpha(1:3,3) % normalised

disp('2. Smoothing: p(h(1)|v(1:3))')
beta=HMMbackward(v,phghm,pvgh);
[phtgV1T,phthtpgV1T]=HMMsmooth(alpha,beta,pvgh,phghm,v);
    % Outputs:
        % phtgV1T : smoothed posterior p(h(t)|v(1:T))
        % phthtpgV1T : smoothed pair p(h(t),h(t+1)|v(1:T))
ph1gV13 = phtgV1T(1:3,1) % p(h(1)|v(1:T)) is all rows 1:3 (1:T), 1st col (h1)
ph3gV13 = phtgV1T(1:3,3) % alternate method of filtering
ph3gV13 == alphaH3; % notice the same

gamma=HMMgamma(alpha,phghm); % alternative alpha-gamma (RTS) method, seems to correspond to smoothed answer

disp('1. Likelihood: p(v(1:3))')
likelihood = exp(loglik) % need to exponentiate to get likelihood

disp('3. Most probable hidden state seq. argmax_h1:3p(h(1:3)|v(1:3)) = Viterbi')

[maxstate logprob]=HMMviterbi(v,phghm,ph1,pvgh); % most likely joint state
maxstate