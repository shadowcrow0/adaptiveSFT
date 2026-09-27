# 用 R 的 sft::sic 對同一份固定資料算一次，存成 json 給 tests/test_sic.py::test_against_r_oracle 逐位元比對。
# 資料由 tests/data/make_sic_input.py 產生（固定種子）。只需要跑一次；json 進版本庫後 pytest 不需要 R。
#   Rscript tests/data/make_sic_oracle.R
suppressMessages(library(sft))
suppressMessages(library(jsonlite))
here <- "tests/data"
d <- read.csv(file.path(here, "sic_input.csv"))
cells <- split(d$rt, d$cell)
HH <- cells$HH; HL <- cells$HL; LH <- cells$LH; LL <- cells$LL
s <- sic(HH = HH, HL = HL, LH = LH, LL = LL, mictest = "art")
aov <- mic.test(HH, HL, LH, LL, method = "anova")
times <- sort(unique(c(HH, HL, LH, LL)))
out <- list(
  cells = list(HH = HH, HL = HL, LH = LH, LL = LL),
  SIC_times = times,
  SIC_values = s$SIC(times),
  Dplus = unname(s$SICtest$positive$statistic), p_Dplus = s$SICtest$positive$p.value,
  Dminus = unname(s$SICtest$negative$statistic), p_Dminus = s$SICtest$negative$p.value,
  dom_stat = s$Dominance$statistic, dom_p = s$Dominance$p.value,
  MIC = unname(s$MICtest$statistic), p_MIC_art = s$MICtest$p.value, p_MIC_anova = aov$p.value
)
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE), file.path(here, "sic_r_oracle.json"))
cat("wrote", file.path(here, "sic_r_oracle.json"), "\n")
