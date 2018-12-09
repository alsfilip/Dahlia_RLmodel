functions{
	real RL_log(int[] chosen,int[] action, int[] sub, int[] trial, vector reward, int N,vector alpha_f, vector beta_f,vector alpha_a, vector beta_a, vector C, vector D){
    matrix[1,2] Qf;
    matrix[1,2] Qa;
    real probf1;
    vector[N] lprob;
    real lr_f;
    real lr_a;
    real sm_f;
    real sm_a;
    real c;
    real c_1;
    real d;
    
    c_1 = 0; // Value of C_1 - C_2
    for(i in 1:N){
      // Transform parameters from real line to their respective supports
      lr_f = 1/(1+exp(-alpha_f[sub[i]]));  // Learning rate for fractals
      sm_f = exp(beta_f[sub[i]]);          // Softmax (sensitivity) for fractals     
      lr_a = 1/(1+exp(-alpha_a[sub[i]]));  // Learning rate for actions
      sm_a = exp(beta_a[sub[i]]);          // Softmax for actions
      c = 1/(1+exp(-C[sub[i]]));           // Sensitivity for previous choice (should be changed to be between 0 and +inf)
      d = 1/(1+exp(-D[sub[i]]));           // Sensitivity for fractal 1 (should be changed to be between 0 and +inf)
      
      if(trial[i] == 1){
        Qf[1,1:2] = [.5,.5];
        Qa[1,1:2] = [.5,.5];
        c_1 = 0;
      }
      // Calculate probability of choosing first fractal
      probf1 = 1/(1+exp(-(sm_f*(Qf[1,1]-Qf[1,2])+sm_a*(Qa[1,1]-Qa[1,2])+c*c_1+d)));
      
      // Get value of (C_1 - C_2) for next trial
      if(chosen[i] == 1){
        lprob[i] = log(probf1);
        c_1 = 1;
      }
      else if(chosen[i] == 2){
        lprob[i] = log(1-probf1);
        c_1 = -1;
      }
      // Update Q values for fractals and actions
      Qf[1,chosen[i]] = Qf[1,chosen[i]]+lr_f*(reward[i]-Qf[1,chosen[i]]);
      Qa[1,action[i]] = Qf[1,action[i]]+lr_a*(reward[i]-Qf[1,action[i]]);
    }
    return sum(lprob);
	}
}
data{
  // Data to be imported
  int N;               // Total number of observations (Nsub*Trials)
  int Nsub;            // Number of subjects (128)
  int trial[N];        // Vector of trial indicies for each subject (1-90)
  int chosen[N];       // Long vector of subject choices of fractals (1 or 2)
  int action[N];       // Long vector of subject actions (1 for left 2 for right)
  int sub[N];          // Long vector of subject indicies (1-128)
  vector[N] reward;    // Long vector of rewards
}
parameters{
  // Hyper parameters to be estimated
  real af_mu;             // Fractal learning rate prior mean
  real<lower=0> af_sig;   // Fractal learning rate prior variance
  real aa_mu;             // Action learning rate prior mean
  real<lower=0> aa_sig;   // Action learning rate prior variance
  real bf_mu;             // Fractal softmax prior mean
  real<lower=0> bf_sig;   // Fractal softmax prior variance
  real ba_mu;             // Action softmax prior mean
  real<lower=0> ba_sig;   // Action softmax prior variance
  real c_mu;              // Perseveration prior mean
  real<lower=0> c_sig;    // Perseveration prior variance
  real d_mu;              // Fractal bias prior mean
  real<lower=0> d_sig;    // Fractal bias prior variance
  
  // Prior parameters (vectors of parameters for each subjects)
  vector [Nsub] alpha_f;  // Fractal learning rate
  vector [Nsub] beta_f;   // Fractal softmax
  vector [Nsub] alpha_a;  // Action learning rate
  vector [Nsub] beta_a;   // Action softmax
  vector [Nsub] C;        // Perseveration
  vector [Nsub] D;        // Fractal Bias
}
model{
  // Model parameters - Hyper parameters
  af_mu ~ normal(0,1);
  af_sig ~ uniform(0,3);
  aa_mu ~ normal(0,1);
  aa_sig ~ uniform(0,3);
  bf_mu ~ normal(0,1);
  bf_sig ~ uniform(0,3);
  ba_mu ~ normal(0,1);
  ba_sig ~ uniform(0,3);
  c_mu~normal(0,1);
  c_sig~uniform(0,2);
  d_mu~normal(0,1);
  d_sig~uniform(0,2);
  
  // Model parameters - all put on the real line to facilitate parameter search
  alpha_f~normal(af_mu,af_sig);
  alpha_a~normal(aa_mu,aa_sig);
  beta_f~normal(bf_mu,bf_sig);
  beta_a~normal(ba_mu,ba_sig);
  C~normal(c_mu,c_sig);
  D~normal(d_mu,d_sig);
  // LL function to maximize (specified above in the "functions" section)
  chosen~RL(action, sub, trial, reward, N, alpha_f, beta_f, alpha_a, beta_a, C, D);
}