functions{
	real RL_log(int[] chosen, int[] unchosen, int[] action, int[] sub, int[] trial, int[] f1_side, int[] f2_side, vector reward, int N,vector alpha_f, vector beta_f,vector alpha_a, vector beta_a, vector C_f, vector C_a, vector D_f, vector D_a){
    matrix[1,2] Qf;
    matrix[1,2] Qa;
    real probf1;
    vector[N] lprob;
    real lr_f;
    real lr_a;
    real sm_f;
    real sm_a;
    real cf_1;
    real ca_1;
    real c_f;
    real c_a;
    real d_f;
    real d_a;
    real da_1;
    
    for(i in 1:N){
      // Transform parameters from real line to their respective supports
      lr_f = 1/(1+exp(-alpha_f[sub[i]]));  // Learning rate for fractals
      sm_f = exp(beta_f[sub[i]]);          // Softmax (sensitivity) for fractals     
      lr_a = 1/(1+exp(-alpha_a[sub[i]]));  // Learning rate for actions
      sm_a = exp(beta_a[sub[i]]);          // Softmax for actions
      c_f = C_f[sub[i]];				           // Persistence - last fractal
      c_a = C_a[sub[i]];                   // Persistence - last action
      d_f = D_f[sub[i]];                   // Bias towards fractal 1
      d_a = D_a[sub[i]];                   // Bias towards action
      
      if(trial[i] == 1){
        Qf[1,1:2] = [.5,.5];   // Q-values for fractals
        Qa[1,1:2] = [.5,.5];   // Q-values for actions
        cf_1 = 0;              // Initial persistence for fractal
        ca_1 = 0;              // Initial persistence for action
      }

      // Trying to capture bias towards one action
      if(f1_side[i] == action[i]){
        da_1 = 1;
      }
      else{
        da_1 = -1;
      }

      // Calculate probability of choosing first fractal
      probf1 = 1/(1+exp(-(sm_f*(Qf[1,1]-Qf[1,2])+sm_a*(Qa[1,f1_side[i]]-Qa[1,f2_side[i]])+c_f*cf_1+c_a*ca_1+d_f+d_a*da_1)));
      
      // Get value of last fractal and last action
      if(chosen[i] == 1){
        lprob[i] = log(probf1);
        cf_1 = 1;
      }
      else if(chosen[i] == 2){
        lprob[i] = log(1-probf1);
        cf_1 = -1;
      }
      if(action[i] == 1){
        ca_1 = 1;
      }
      else if(action[i] == 2){
        ca_1 = -1;
      }
      // Update Q values for fractals and actions
      Qf[1,chosen[i]] = Qf[1,chosen[i]]+lr_f*(reward[i]-Qf[1,chosen[i]]);
      Qf[1,unchosen[i]] = 1-Qf[1,chosen[i]];
      Qa[1,action[i]] = Qa[1,action[i]]+lr_a*(reward[i]-Qa[1,action[i]]);
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
  int unchosen[N];     // Long vector of unchosen fractal (1 or 2)
  int action[N];       // Long vector of subject actions (1 for left 2 for right)
  int f1_side[N];      // Long vector with side on which Fractal 1 appeared
  int f2_side[N];      // Long vector with side on which Fractal 2 appeared
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
  real cf_mu;             // Perseveration prior mean
  real<lower=0> cf_sig;   // Perseveration prior variance
  real ca_mu;             // Perseveration prior mean
  real<lower=0> ca_sig;   // Perseveration prior variance
  real df_mu;             // Fractal bias prior mean
  real<lower=0> df_sig;   // Fractal bias prior variance
  real da_mu;             // Fractal bias prior mean
  real<lower=0> da_sig;   // Fractal bias prior variance
  
  // Prior parameters (vectors of parameters for each subjects)
  vector [Nsub] alpha_f;  // Fractal learning rate
  vector [Nsub] beta_f;   // Fractal softmax
  vector [Nsub] alpha_a;  // Action learning rate
  vector [Nsub] beta_a;   // Action softmax
  vector [Nsub] C_f;      // Persistance - fractal
  vector [Nsub] C_a;      // Persistance - action
  vector [Nsub] D_f;      // Fractal Bias
  vector [Nsub] D_a;      // Action Bias
}
model{
  // Model parameters - Hyper parameters
  af_mu ~ normal(0,1);
  af_sig ~ uniform(0,5);
  aa_mu ~ normal(0,1);
  aa_sig ~ uniform(0,5);
  bf_mu ~ normal(0,1);
  bf_sig ~ uniform(0,5);
  ba_mu ~ normal(0,1);
  ba_sig ~ uniform(0,5);
  cf_mu~normal(0,1);
  cf_sig~uniform(0,5);
  ca_mu~normal(0,1);
  cf_sig~uniform(0,5);
  df_mu~normal(0,1);
  df_sig~uniform(0,5);
  da_mu~normal(0,1);
  da_sig~uniform(0,5);
  
  // Model parameters - all put on the real line to facilitate parameter search
  alpha_f~normal(af_mu,af_sig);
  alpha_a~normal(aa_mu,aa_sig);
  beta_f~normal(bf_mu,bf_sig);
  beta_a~normal(ba_mu,ba_sig);
  C_f~normal(cf_mu,cf_sig);
  C_a~normal(ca_mu,ca_sig);
  D_f~normal(df_mu,df_sig);
  D_a~normal(da_mu,da_sig);
  
  // LL function to maximize (specified above in the "functions" section)
  chosen~RL(unchosen, action, sub, trial, f1_side, f2_side, reward, N, alpha_f, beta_f, alpha_a, beta_a, C_f, C_a, D_f, D_a);
}