#!/usr/bin/env bash
# 在這個 cloud VM 上把 adaptiveSFT 的 R 依賴裝起來，完全不碰 CRAN
# （CRAN 被環境的 Trusted 網路等級擋住；apt 的 archive.ubuntu.com 和 GitHub 沒被擋）。
# 實測 2026-09-24：Ubuntu 24.04、R 4.3.3，全程約 5 分鐘。
# 只裝套件，不編譯任何 .stan、不修改 repo 裡的 R 檔。
set -euo pipefail

# 1. Ubuntu 有打包的：rstan 2.32.5（StanHeaders 2.32.5，stanc 2.32 → lnrm2.stan 舊語法仍可編）
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  r-cran-rstan r-cran-suppdists r-cran-statmod \
  r-cran-rcurl r-cran-pcapp r-cran-ks r-cran-colorspace r-cran-locfit \
  r-cran-ggplot2 r-cran-rcolorbrewer

# 2. Ubuntu 沒打包的：從 GitHub 的 CRAN 唯讀鏡像抓原始碼，按依賴順序 R CMD INSTALL
#    sft → fda → fds → rainbow → hdrcde → ash ;  diffIRT → statmod(已裝)
work=$(mktemp -d)
for p in ash hdrcde rainbow fds fda sft diffIRT; do
  git clone -q --depth 1 "https://github.com/cran/$p.git" "$work/$p"
  R CMD INSTALL --no-docs --no-multiarch "$work/$p" > "$work/install_$p.log" 2>&1 \
    || { echo "$p FAILED"; tail -20 "$work/install_$p.log"; exit 1; }
  echo "$p OK"
done

# 3. 驗證
Rscript -e 'cat("rstan", requireNamespace("rstan",quietly=TRUE),
                " diffIRT", requireNamespace("diffIRT",quietly=TRUE),
                " sft", requireNamespace("sft",quietly=TRUE), "\n")'
