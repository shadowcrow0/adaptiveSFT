# Oracle for adaptivesft/ddm.py: diffIRT::simdiffT run at the parameter values the original scripts use.
# R and Python cannot share a random stream, so the comparison is distributional (P(x = 1), RT mean and
# quantiles over 20000 trials), plus the closed form implied by simdiffT.r:6.
#   Rscript tests/data/make_ddm_oracle.R
suppressMessages(library(diffIRT))
suppressMessages(library(jsonlite))
here <- "tests/data"
set.seed(20260930)
cases <- list(
  # psiSimulation_functions.R:96-99 at the top of the colour range (scaled intensity 1 -> drift = v)
  list(name = "psi_range_top", a = 1.45, mv = 1.6, sv = 0.25, ter = 0.1),
  # the same observer under the 'threshold' reading of a (decisions_for_author.md A): separation 2a
  list(name = "psi_range_top_2a", a = 2.90, mv = 1.6, sv = 0.25, ter = 0.1),
  # simulateLNRM_ogival.R:32-35 at scaled intensity 0.5
  list(name = "lnrm_mid", a = 3.0, mv = 1.0, sv = 0.2, ter = 0.1),
  # the low-salience level the 2019 Psi script hard-codes (26MAR2019.R:211): x = 16.45 -> scaled 0.2375
  list(name = "psi2019_low_colour", a = 1.45, mv = 0.2375 * 1.6, sv = 0.25, ter = 0.1)
)
out <- list()
for (cs in cases) {
  s <- simdiffT(20000, cs$a, cs$mv, cs$sv, cs$ter)
  out[[cs$name]] <- list(a = cs$a, mv = cs$mv, sv = cs$sv, ter = cs$ter,
                         p_upper = mean(s$x), rt_mean = mean(s$rt),
                         rt_q = unname(quantile(s$rt, c(.1, .25, .5, .75, .9))),
                         closed_form_p = exp(cs$a * cs$mv) / (1 + exp(cs$a * cs$mv)))   # simdiffT.r:6
  cat(sprintf("%-20s P(x=1)=%.4f  closed form %.4f  mean rt %.4f\n", cs$name, out[[cs$name]]$p_upper,
              out[[cs$name]]$closed_form_p, out[[cs$name]]$rt_mean))
}
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE), file.path(here, "ddm_r_oracle.json"))
cat("wrote", file.path(here, "ddm_r_oracle.json"), "\n")
