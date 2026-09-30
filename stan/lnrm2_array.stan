// lnrm2.stan with the five array declarations rewritten in the syntax Stan >= 2.33 requires
// (issue.md S1). Nothing else changed: same data, parameters, priors and likelihood.
// Used only by tests/data/make_lnrm_oracle.R to produce a Stan oracle for adaptivesft/models.py.
// The original lnrm2.stan in the repo root is left untouched.
data {
   int<lower=1> N;
   array[N] real intensity;
   array[N] int<lower=0,upper=1> correct;
   real<lower=0> minRT;
   array[N] real<lower=0> rt;
}
transformed data {
   array[N] real square_intensity;
   square_intensity = square(intensity);
}
parameters{
   real alpha;
   real alpha2;
   real mu;
   real<lower=0> varZ;
   real<lower=0,upper=minRT> psi;
}
transformed parameters {
   array[2, N] real z;

   for (tr in 1:N) {
      z[1,tr] = mu - alpha * intensity[tr] - alpha2 * square_intensity[tr];
      z[2,tr] = mu + alpha * intensity[tr] + alpha2 * square_intensity[tr];
   }
}
model {
   varZ ~ inv_gamma(1,.1);
   mu ~ normal(0,1);
   alpha ~ normal(0,2);
   alpha2 ~ normal(0,1);

   // psi has improper flat prior on positive reals
   for ( tr in 1:N) {
      if ( correct[tr] ) {
         target += lognormal_lpdf(rt[tr] - psi | z[1,tr], varZ);
         target += lognormal_lccdf(rt[tr] - psi | z[2,tr], varZ);
      }
      else {
         target += lognormal_lpdf(rt[tr] - psi | z[2,tr], varZ);
         target += lognormal_lccdf(rt[tr] - psi | z[1,tr], varZ);
      }
   }
}
