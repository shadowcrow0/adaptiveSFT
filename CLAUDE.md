# CLAUDE.md

## 輸出風格

**簡潔模式 (concise)。** 直接給結論，不要鋪陳、不要重述問題、
不要在結尾摘要已經說過的話。能用一句講完就不要用一段。

**解釋概念時用 ASCII 圖。** 說明流程、資料結構、模型行為、或任何有
「東西 A 怎麼變成東西 B」的內容時，畫一張 ASCII 圖，不要只用文字描述。

只用 ASCII。不要用 Unicode 數學符號（psi、sigma、Phi 等一律寫英文，
或直接用程式碼裡的變數名），也不要用 LaTeX——閱讀環境包含 GitHub 手機 app，
兩者都不會正確顯示。

範例：

```
資料 (intensity, rt, correct)
      |
      v
  lnrm2.stan  --> 後驗 {mu, alpha, alpha2, varZ, psi}
      |
      v
  解二次式 --> high / low salience
```

```
   z[1,tr] = mu - d     "答對" 累積器
                 ^
                 |  d = alpha*intensity + alpha2*intensity^2
                 v
   z[2,tr] = mu + d     "答錯" 累積器
```

## 撰寫文件時

區分「程式碼字面寫了什麼」與「從程式碼推導出來的」，並明確標示。
引用程式碼一律附 `檔名:行號`，且寫入前要比對過實際檔案內容。
