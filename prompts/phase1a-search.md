# Phase 1a: 搜尋 + 深讀 + 逐字抄錄

**輸入：** `workspace/phase0-plan.md` + `workspace/coverage.chk`
**輸出：** `workspace/search-results/Q{n}/S{id}.md` + `workspace/source-registry.md` + `workspace/execution-log.md`
**完成條件：** 所有搜尋結果已存檔，source-registry 已建立 → 進入 Phase 1b

**本 Phase 只做 4 件事：**
1. 根據 coverage gap 產生 query
2. 搜尋並記錄來源
3. 深讀並逐字抄錄原文
4. 將所有結果存入 workspace 檔案

---

## 搜尋引擎

**核心（每次必用，透過直接 HTTP API 呼叫）：**

| 引擎 | 用途 |
|------|------|
| Brave Search API (EN) | 英文獨立索引 |
| Serper API (gl=tw, hl=zh-TW) | 繁體中文 Google |
| Serper API (EN) | 英文 Google |

**學術（技術主題追加）：**
Serper API + `site:semanticscholar.org` 或 `site:arxiv.org`

**擴展語言（依主題）：** 讀取 `prompts/ref-multilingual.md`

**中國資訊：** Serper (gl=cn, hl=zh-CN) + Baidu Search API 同時搜

---

## Step 1: 讀取計畫 + 初始化

1. 讀取 `workspace/phase0-plan.md`（重點讀結構化 Header 和子問題 DAG）
2. 讀取 `workspace/coverage.chk`
3. 建立目錄：`workspace/search-results/Q1/`、`Q2/` ... 等
4. 初始化 `workspace/source-registry.md`：

```markdown
# Source Registry

| source_id | url | title | fetched_title | tier | url_status | date | engines | roles | subquestion |
|-----------|-----|-------|--------------|------|------------|------|---------|-------|-------------|
```

5. 初始化 `workspace/execution-log.md`：

```markdown
# 執行日誌
**研究主題：** {主題}
**開始時間：** {timestamp}
**搜尋預算：** 0 / {total}

## 已搜 Query 清單

## 第 1 輪
```

---

## Step 2: 生成第一輪 Query

**漸進式 query 生成（不一次展開所有 query）：**

第一輪，每個子問題只生成最小集：
- advocate: 1 個 query family（EN + ZH 版本）
- critic: 1 個 query family（EN + ZH 版本）
- perspective: 0-1 個（有明確視角時）
- academic: 0-1 個（學術/技術主題時）

規則：
- 每個 query 5-10 詞
- 正反方 query 必須有明顯差異
- 對照 execution-log 的已搜 query 清單做語義去重
- 生成後立即追加到 execution-log

後續輪次的 query 由 Phase 1b 的 coverage gap 和 challenge checklist 觸發。

---

## Step 3: 平行搜尋

按 DAG 順序。獨立子問題在同一 message 平行搜尋。

每個 query family 的搜尋組合：
```
EN query  → WebSearch + Brave                          (2 次)
ZH query  → WebSearch + Serper(gl=tw, hl=zh-TW)        (2 次)
學術      → Serper(site:arxiv.org/semanticscholar.org)  (1-2 次)
擴展語言  → 依 ref-multilingual.md                      (依主題)
```

**每次搜尋後，更新 execution-log 的搜尋計數。**

---

## Step 4: URL 優先級排序

合併去重所有搜尋結果，按以下因子評分：

| 因子 | 加分 |
|------|------|
| 跨引擎命中 | +2/engine |
| 跨角色命中（正+反都出現） | +5 |
| Domain Authority（T1-T2: +3, T3: +2, T4: +1） | 依層級 |
| 時效性（依 freshness SLA） | +3/+2/+1 |

**深讀配額（每子問題每輪）：**
- advocate: 2 篇
- critic: 2 篇
- perspective: 1 篇
- overflow: 最多 2 篇（僅在前 5 篇仍有關鍵缺口時）

**禁止因總排序把 critic 擠到 0 篇。**

---

## Step 5: URL 活性驗證

所有待深讀的 URL → 用 Bash 呼叫 urlhealth CLI（零例外）：

```bash
echo '{"urls":["URL1","URL2",...]}' | .venv/bin/python3 grounding_scripts/urlhealth.py --cli
```

| 狀態 | 動作 |
|------|------|
| LIVE | 繼續深讀 |
| STALE | 可引用，附 Wayback URL |
| LIKELY_HALLUCINATED | **立即移除** |
| UNKNOWN | 用 Serper Scrape API 重試一次 |

---

## Step 6: 深讀 + 逐字抄錄

**三階梯抓取（依序嘗試）：**
1. **WebFetch**（首選，保留完整結構）
2. **Serper Scrape API**（備援，純文字）
3. 兩者皆失敗 → 標記 URL 為 `[UNREACHABLE]`，記入 `workspace/gap-log.md`，不得僅依賴搜尋摘要寫入 QUOTE/NUMBER

**Bedrock API Throttling 處理：** 若遇到 429 或 throttling，等待 3 秒後重試，最多 3 次。持續失敗 → 讀取 fallback.md。

**逐字抄錄規則：**

1. 找到與子問題相關的段落
2. 逐字複製**最多 3 個關鍵句**（選最有證據力的）
3. 數字必須連同原文完整句子一起複製
4. 固定格式（每個 quote/number 有唯一 ID）：

```
QUOTE[S{id}-Q{n}]: "{從原文逐字複製的句子}" — {url}
NUMBER[S{id}-N{n}]: {數字} {單位} — Original: "{含該數字的原文完整句子}" — {url}
```

**禁止：**
- ❌ 讀完原文後用自己的話重述
- ❌ 合併多個數字做計算（除非原文自己算了）
- ❌ 只讀 WebFetch 摘要就下結論

---

## Step 7: 存入 Workspace

**7a. 每個來源存為獨立檔案** `workspace/search-results/Q{n}/S{id}.md`：

```markdown
# Source S{id}: {title}

- URL: {url}
- Fetched Title: {實際頁面標題}
- URL Status: {LIVE/STALE}
- Tier: T{n}
- Fetch Date: {YYYY-MM-DD}
- Engines: {brave, serper_tw, ...}
- Role: {advocate/critic/perspective:{name}}
- Subquestion: Q{n}

## Verbatim Quotes
QUOTE[S{id}-Q1]: "{原文}"
QUOTE[S{id}-Q2]: "{原文}"
NUMBER[S{id}-N1]: {數字} — Original: "{原句}"
```

**7b. 更新 source-registry.md**（追加一行）

**7c. 更新 execution-log.md**

---

## Gate Check（必須全過才能離開 Phase 1a）

```
□ 1. 所有深讀的來源都有獨立檔案（workspace/search-results/Q{n}/S{id}.md）？
□ 2. source-registry.md 已更新？
□ 3. execution-log.md 搜尋計數正確？
□ 4. 所有 QUOTE/NUMBER 都有唯一 ID？
□ 5. 所有 UNREACHABLE URL 已記入 gap-log.md？
□ 6. coverage.chk 已更新搜尋進度？
→ 全部 ✅ → 進入 Phase 1b
```
