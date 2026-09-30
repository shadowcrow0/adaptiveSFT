# Oracle for adaptivesft/psi.py: the Psi loop of psiSimulation_functions.R (Est.Trial.Psi.Color, :5-168),
# copied line by line, with one change so that R and Python see the same observer:
# instead of simdiffT (:104-106) the response is drawn from pm.function with a pre-drawn uniform u[trial]
# (tests/data/psi_oracle_input.csv, made by make_psi_oracle_input.py). Everything else is the original code.
#   Rscript tests/data/make_psi_oracle.R
suppressMessages(library(jsonlite))
here <- "tests/data"
inp <- read.csv(file.path(here, "psi_oracle_input.csv"))
u <- inp$u
nTrials <- min(length(u), as.integer(Sys.getenv("PSI_ORACLE_TRIALS", "40")))   # the verbatim R loops are slow (~750k iterations per trial)

pm.function <- function (x,a,b,d) .5 * d + (1-d) * pnorm(x,a,b)            # :3
inv.pm.function <- function (y,a,b,d) qnorm((y-.5*d)/(1-d), a, b)         # :171

prior <- NA
sim.a <- 6                                                                 # :10-12
sim.b <- 15
sim.d <- .01
x.range <- c(-55,50); x.step <- 1                                          # :15-18
a.range <- c(-25,45); a.step <- 1
b.range <- c(1,50); b.step <- 1
d <- .01

x <- seq(x.range[1],x.range[2],x.step)                                     # :24-27
a <- seq(a.range[1],a.range[2],a.step)
b <- seq(b.range[1],b.range[2],b.step)
r <- c(0,1)

pR.LX <- array(dim = c(length(r), length(a), length(b), length(x)))        # :31-40
for (i in 1:length(r)) for (j in 1:length(a)) for (k in 1:length(b)) for (l in 1:length(x))
  pR.LX[i,j,k,l] <- (1-r[i]) + (2*r[i]-1) * pm.function(x[l],a[j],b[k],d)

pL <- array(data = 1/(length(a)*length(b)), dim = c(1,length(a),length(b),1))   # :45-46

pR.X <- array(dim = c(length(r), 1, 1, length(x)))                         # :52-57
for (i in 1:length(r)) for (j in 1:length(x)) pR.X[i,1,1,j] <- sum(pR.LX[i,,,j] * pL[1,,,1])

pL.XR <- array(dim = c(length(r), length(a), length(b), length(x)))        # :60-69
for (i in 1:length(r)) for (j in 1:length(a)) for (k in 1:length(b)) for (l in 1:length(x))
  pL.XR[i,j,k,l] <- pL[1,j,k,1] * pR.LX[i,j,k,l] / pR.X[i,,,l]

entropy.XR <- array(dim = c(length(r), 1, 1, length(x)))                   # :72-77
for (i in 1:length(r)) for (j in 1:length(x)) entropy.XR[i,1,1,j] <- -sum(pL.XR[i,,,j] * log10(pL.XR[i,,,j]))

expected.entropy.X <- array(dim = c(1, 1, 1, length(x)))                   # :80-83
for (i in 1:length(x)) expected.entropy.X[1,1,1,i] <- sum(entropy.XR[,1,1,i] * pR.X[,1,1,i])

next.intensity.index <- which.min(expected.entropy.X[1,1,1,])              # :86-87
next.intensity <- x[next.intensity.index]

intens <- numeric(nTrials); resp <- integer(nTrials); a.est.v <- numeric(nTrials); b.est.v <- numeric(nTrials)
for (trial in 1:nTrials) {                                                 # :102
  intens[trial] <- next.intensity
  sim.response <- as.integer(u[trial] < pm.function(next.intensity, sim.a, sim.b, sim.d))   # replaces :104-106
  resp[trial] <- sim.response

  pL <- pL.XR[sim.response+1, , , next.intensity.index]                    # :112-114
  dim(pL) <- c(1, dim(pL)[1], dim(pL)[2], 1)

  for (i in 1:length(r)) for (j in 1:length(x)) pR.X[i,1,1,j] <- sum(pR.LX[i,,,j] * pL[1,,,1])   # :117-121
  for (i in 1:length(r)) for (j in 1:length(a)) for (k in 1:length(b)) for (l in 1:length(x))    # :124-132
    pL.XR[i,j,k,l] <- pL[1,j,k,1] * pR.LX[i,j,k,l] / pR.X[i,,,l]
  for (i in 1:length(r)) for (j in 1:length(x)) entropy.XR[i,1,1,j] <- -sum(pL.XR[i,,,j] * log10(pL.XR[i,,,j]))  # :135-139
  for (i in 1:length(x)) expected.entropy.X[1,1,1,i] <- sum(entropy.XR[,1,1,i] * pR.X[,1,1,i])  # :142-144

  next.intensity.index <- which.min(expected.entropy.X[1,1,1,])            # :149-150
  next.intensity <- x[next.intensity.index]

  a.est <- 0; b.est <- 0                                                   # :154-163
  for (i in 1:length(a)) a.est <- a.est + sum(a[i] * pL[1,i,,1])
  for (j in 1:length(b)) b.est <- b.est + sum(b[j] * pL[1,,j,1])
  a.est.v[trial] <- a.est; b.est.v[trial] <- b.est
}

n <- nTrials
out <- list(
  sim = list(a = sim.a, b = sim.b, d = sim.d),
  intensity = intens, response = resp, alpha = a.est.v, beta = b.est.v,
  high = inv.pm.function(.99, a.est.v[n], b.est.v[n], sim.d),              # :183-184
  low  = inv.pm.function(.90, a.est.v[n], b.est.v[n], sim.d)
)
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE), file.path(here, "psi_r_oracle.json"))
cat("wrote", file.path(here, "psi_r_oracle.json"), "trials", n, "final alpha", a.est.v[n], "beta", b.est.v[n], "\n")
