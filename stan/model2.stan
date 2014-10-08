data {
  int n;
  vector[2] y[n];
  vector[2] mu_mean;
  cov_matrix[2] mu_cov;
  vector<lower=0>[2] sigma_mean;
  vector<lower=0>[2] sigma_var;
  vector<lower=0>[2] tau_mean;
  vector<lower=0>[2] tau_var;
  real<lower=0> rho_a;
  real<lower=0> rho_b;  
}
parameters {
  vector[2] mu[n];
  vector<lower=0>[2] tau;
  vector[2] sigma;
  real<lower=0, upper=1> rho;
}
transformed parameters {
  cov_matrix[2] Sigma;
  cholesky_factor_cov[2] Sigma_chol;
  {
    matrix[2, 2] tmp;
    tmp[1, 1] <- 1.0;
    tmp[2, 2] <- 1.0;
    tmp[1, 2] <- rho;
    tmp[2, 1] <- rho;
    Sigma <- quad_form_diag(tmp, sigma);
    Sigma_chol <- cholesky_decompose(Sigma);
  }
}
model {
  rho ~ beta(rho_a, rho_b);
  for (i in 1:2) {
    tau[i] ~ gamma(pow(tau_mean[i], 2) / tau_var[i],
		   tau_mean[i] / tau_var[i]);
  }
  for (i in 1:2) {
    sigma[i] ~ gamma(pow(sigma_mean[i], 2) / sigma_var[i],
		   sigma_mean[i] / sigma_var[i]);
  }
  mu[1] ~ multi_normal(mu_mean, mu_cov);
  for (i in 2:n) {
    mu[i, 1] ~ normal(mu[i - 1, 1], tau[1]);
    mu[i, 2] ~ normal(mu[i - 1, 2], tau[2]);
  }
  for (i in 1:n) {
    y[i] ~ multi_normal_cholesky(mu[i], Sigma_chol);
  }
}
generated quantities {
  vector[2] yhat;
  yhat <- multi_normal_cholesky_rng(mu[n], Sigma_chol);
}

