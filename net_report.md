# 網路連線調查報告

## 結論

- **這台機器不在大學網路上。** 它是 Anthropic 管理的雲端沙箱 VM（`hostname` = `vm`，
  內網 IP `192.0.2.2`，屬於文件保留網段 TEST-NET-1，不是任何真實校園網路的位址）。
  使用者只是透過瀏覽器/Claude Code 操作它。
- **決定能連什麼的不是校園網路規則，是這個 session 的 Network access 等級
  （目前 = Trusted）底下的 egress allowlist。**
- `.edu` 網域和所有 CRAN mirror 都不在 allowlist 裡，一律被上游 gateway 回
  `403`；`pypi.org`、`archive.ubuntu.com` 等套件源在 allowlist 裡，所以正常。

## 連線路徑

```
瀏覽器 (使用者)
     │  https / websocket
     ▼
Anthropic 雲端 VM  (hostname=vm, 內網 192.0.2.2/24, gw 192.0.2.1)
     │  本機程序發出的 HTTPS 請求
     ▼
agent proxy  127.0.0.1:46653   (HTTPS_PROXY, TLS 在此重新終止)
     │  CONNECT host:443
     ▼
egress gateway  (org 的 policy-enforcing 出口代理)
     │
     ├── host 在 allowlist  → 放行 → 真正的 internet
     └── host 不在 allowlist → 回應 403 → curl: (56) CONNECT tunnel failed
```

`/root/.ccr/README.md` 開頭即說明：

> Outbound HTTPS from this session goes through a local proxy at
> http://127.0.0.1:46653 (set via HTTPS_PROXY) which tunnels to a
> policy-enforcing egress proxy. TLS is re-terminated there...

以及對 403 的定義：

> ### 403 / 407 from the proxy
> The destination host is not allowed by your organization's egress policy
> for this session. Do not retry or route around it — report the blocked
> host.

也就是說 403 一律是「這個 session 的組織出口政策不允許」，不是網路故障、
不是校園防火牆，也不該重試或繞過。

## 證據

### 1. 機器身分

```
$ hostname
vm

$ cat /etc/hostname
vm

$ cat /etc/resolv.conf
nameserver 8.8.8.8
nameserver 8.8.4.4
options timeout:2 attempts:3

$ hostname -I          # 沒有 `ip` 指令，改用這個
192.0.2.2

$ cat /proc/net/route   # default gw = 192.0.2.1 (解出 010200C0 → little-endian)
Iface  Destination  Gateway   Flags ...
eth0   00000000     010200C0  0003  ...   # default route
eth0   000200C0     00000000  0001  ...   # 192.0.2.0/24 on-link

$ cat /etc/os-release | head -3
PRETTY_NAME="Ubuntu 24.04.4 LTS"
NAME="Ubuntu"
VERSION_ID="24.04"

$ cat /etc/hosts
127.0.0.1 localhost
160.79.104.10 api.anthropic.com   # 寫死指到 Anthropic 的 IP
127.0.0.1 runsc                    # runsc = gVisor sandbox runtime 的痕跡
127.0.0.1 vm
```

`192.0.2.0/24` 是 RFC 5737 保留給「文件範例」用的 TEST-NET-1 網段，任何真實
網路都不會把它分配給機器 — 這是容器/沙箱內部虛擬網卡常見的假位址，進一步
說明這不是接在某個真實機構（更不是大學）網路上的實體網卡。

`env | grep -iE "proxy|ccr|ccpool|env_"`（節錄，已對可疑欄位標記，
本身無 token 明碼）：

```
CCR_AGENT_PROXY_ENABLED=1
CCR_UPSTREAM_PROXY_ENABLED=1
CCR_EGRESS_GATEWAY_ENABLED=1
HTTPS_PROXY=http://127.0.0.1:46653
https_proxy=http://127.0.0.1:46653
NO_PROXY=localhost,127.0.0.1,...,api.anthropic.com,...,pypi.org,files.pythonhosted.org,...
ANT_IMAGE_REPOSITORY=sandbox-ccr-default
GH_TOKEN=proxy-injected
GITHUB_TOKEN=proxy-injected
AWS_ACCESS_KEY_ID=proxy-injected
AWS_SECRET_ACCESS_KEY=proxy-injected
CLOUDSDK_AUTH_ACCESS_TOKEN=proxy-injected
```

`ANT_IMAGE_REPOSITORY=sandbox-ccr-default`、`CCR_*` 一系列變數、以及所有
雲端服務憑證都標示 `proxy-injected`，是 Anthropic 沙箱基礎設施的標準命名，
不是任何大學 IT 系統會有的東西。

### 2. .edu 的痕跡

```
$ grep -ri "\.edu" /etc/resolv.conf /etc/hosts
(無輸出，exit code 1 — 完全沒有 .edu 相關字串)

$ python3 -c "import socket; print(socket.gethostbyaddr('192.0.2.1'))"
socket.herror: [Errno 1] Unknown host   # gateway 192.0.2.1 沒有 PTR 記錄
```

`resolv.conf`、`/etc/hosts`、預設閘道反解，三個地方都找不到任何 `.edu` 或
校園網路的痕跡。

`agent proxy` 狀態（`curl -sS "$HTTPS_PROXY/__agentproxy/status"`，已省略
與本題無關欄位）：

```json
{
  "enabled": true,
  "port": 46653,
  "noProxy": "localhost,127.0.0.1,...,api.anthropic.com,...,
              registry.npmjs.org,jsr.io,npm.jsr.io,pypi.org,
              files.pythonhosted.org,index.crates.io,proxy.golang.org,...",
  "recentRelayFailures": [
    {"host": "cran.rstudio.com:443",        "kind": "connect_rejected",
     "detail": "gateway answered 403 to CONNECT (policy denial or upstream failure)"},
    {"host": "packagemanager.posit.co:443", "kind": "connect_rejected", "...": "403"},
    {"host": "cloud.r-project.org:443",     "kind": "connect_rejected", "...": "403"},
    {"host": "ppa.launchpadcontent.net:443","kind": "connect_rejected", "...": "403"}
  ]
}
```

`noProxy` 是「不必走 agent proxy、可直連」的白名單（api.anthropic.com、
npm/pypi/crates/go 等套件登記處），跟能不能連到 CRAN 是不同機制 —
`recentRelayFailures` 才是真正的證據：**在使用者過去嘗試連線時，gateway
已經對 cran.rstudio.com、packagemanager.posit.co、cloud.r-project.org 回過
403**，寫著 "policy denial"。這代表使用者先前的連線失敗，本來就是政策擋下
的，不是網路不通。

### 3. Reachability matrix

指令樣式：
```
curl -sS -m 15 -o /dev/null -w '%{http_code} %{remote_ip}\n' https://HOST/
```

| host | 結果 | 原因 |
|---|---|---|
| cloud.r-project.org | `curl: (56) CONNECT tunnel failed, response 403` | 不在 allowlist，policy denial |
| cran.r-project.org | 同上，403 | 不在 allowlist |
| cran.rstudio.com | 同上，403 | 不在 allowlist |
| packagemanager.posit.co | 同上，403 | 不在 allowlist |
| cran.csie.ntu.edu.tw（台大 CRAN mirror） | 同上，403 | 不在 allowlist（也是 .edu.tw） |
| ftp.ntu.edu.tw | 同上，403 | 不在 allowlist（.edu.tw） |
| www.ntu.edu.tw | 同上，403 | 不在 allowlist（.edu.tw） |
| www.mit.edu | 同上，403 | 不在 allowlist（.edu） |
| arxiv.org | 同上，403 | 不在 allowlist |
| pypi.org | `200 151.101.192.223` | 在 allowlist（Trusted 等級預設含 pip 套件源） |
| files.pythonhosted.org | `404 151.101.128.223`（有連上，404 是路徑問題非阻擋） | 在 allowlist |
| archive.ubuntu.com | `200`（remote_ip 127.0.0.1，TLS 在 proxy 端重終止） | 在 allowlist（apt 套件源） |
| security.ubuntu.com | `301` | 在 allowlist |
| github.com | `400`（有連上，400 是根路徑無回應內容） | 在 allowlist |
| raw.githubusercontent.com | `301` | 在 allowlist |
| api.anthropic.com | `404`，remote_ip `160.79.104.10` | 在 noProxy 白名單，直連 Anthropic |
| www.google.com | `curl: (56) CONNECT tunnel failed, response 403` | 不在 allowlist |
| ifconfig.me | `curl: (56) CONNECT tunnel failed, response 403` | 不在 allowlist（IP 回顯服務會洩漏出口 IP，被擋） |
| api.ipify.org | `curl: (56) CONNECT tunnel failed, response 403` | 不在 allowlist，同上 |

（`archive.ubuntu.com`、`github.com` 等 `remote_ip` 顯示 `127.0.0.1`，是因
為 agent proxy 在本機重新終止 TLS 再轉發，curl 看到的連線端點就是本機
proxy，不代表沒連上 — `http_code` 有實際數值即代表已經連到目的站。）

## 為什麼是這個結果

能連上的清單剛好是 `pypi.org`、`files.pythonhosted.org`、
`archive.ubuntu.com`、`security.ubuntu.com`、`github.com`、
`raw.githubusercontent.com`、`api.anthropic.com` — 全部是「語言/套件生態系
的官方登記處」（pip、apt、git/GitHub）或 Anthropic 自己的 API。這是一份**
典型的「開發環境套件源」allowlist**，跟教育網路完全無關。

CRAN（R 的套件源）、`.edu` 學術網站、Google、IP 回顯服務全部不在這份清單
裡，一律被 gateway 擋在 `403`。這代表：

```
連得到 = 在 Trusted 等級的 egress allowlist 裡
連不到 = 不在，不管是不是「看起來很正經」的學術網站
```

跟「這台機器在不在大學網路」無關 —— 它根本不在大學網路，是雲端 VM；
差別純粹是這個 session 的 **Network access 等級**（目前是 **Trusted**，只
放行固定的一批套件生態系登記處）決定了什麼 host 進得去。

## 怎麼改

1. 到 **claude.ai/code** 的 **環境設定（Environment）** 選單。
2. 找到 **Network access**，把等級從 **Trusted** 改成：
   - **Custom**：手動把 `cloud.r-project.org`（或其他 CRAN mirror）加進
     allowlist，最小權限、只開需要的 host；或
   - **Full**：完全開放出口，不再過濾，但風險面最大。
3. **改完要開一個新的 session** 才會生效 — 目前這個 session 的 proxy 設定
   已經在容器啟動時固化，`/root/.ccr/README.md` 也明講不要嘗試繞過（不能
   改 proxy 設定、不能 unset `HTTPS_PROXY`、不能關 TLS 驗證）。
4. **終端機/session 內部改不了**：allowlist 是 gateway 端（org 層級的出口
   政策）強制的，session 裡沒有任何指令、環境變數、設定檔能繞過或放寬它 —
   這正是 README 裡「不要 retry、不要 route around」這句話要說的事。
