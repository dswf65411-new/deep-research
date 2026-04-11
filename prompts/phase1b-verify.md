# Phase 1b: 驗證 + 條件式辯證 + 迭代

**輸入：** `workspace/search-results/` + `workspace/source-registry.md` + `workspace/coverage.chk`
**輸出：** `workspace/claim-ledger.md` + `workspace/grounding-results/` + 更新 `workspace/coverage.chk` + `workspace/gap-log.md`
**完成條件：** 通過結構化停止條件

**本 Phase 分兩子階段：**
- **1b-A：** Grounding + 品質評估 → 全過就直接進 Phase 2
- **1b-B（僅 1b-A 未過時觸發）：** Subagent 核對 + 辯證 + 補搜 → 迭代

---

## 驗證工具（CLI 模式）

**完整 CLI 用法參見 `prompts/ref-cli-tools.md`**

```
PY=.venv/bin/python3
GS=grounding_scripts
MINICHECK_PY=.minicheck-venv/bin/python3
```

| 工具 | CLI 指令 | 角色 | 何時用 |
|------|---------|------|--------|
| Bedrock Grounding | `echo '{...}' \| $PY $GS/bedrock-guardrails.py --cli` | **主要驗證** | 每個 claim 必跑 |
| MiniCheck | `echo '{...}' \| $MINICHECK_PY $GS/minicheck.py --cli` | **備用** | 僅當 Bedrock 錯誤時 |
| NeMo Grounding | `echo '{...}' \| $PY $GS/nemo-guardrails.py --cli` | **第三備用** | 僅當 Bedrock + MiniCheck 都錯誤時 |

**Bedrock 判定規則（按 claim 類型分門檻）：**

| Claim 類型 | Bedrock 門檻 | 額外要求 |
|-----------|-------------|---------|
| 數字型 / 精確引用 | >= 0.8 | Citation API = precise + URL = LIVE/STALE |
| 比較型 / 排名型 | >= 0.75 | 至少 2 個獨立來源 |
| 因果型 / 預測型 | >= 0.75 | 至少 1 個 primary + 1 個 secondary source |
| 背景定性型 | >= 0.7 | — |

**Bedrock API Throttling：** 遇到 429 → 等待 3 秒重試，最多 3 次。持續失敗 → 改用 MiniCheck。
**批次策略：** 同子問題的 claims 盡量用一次 Bash 呼叫打包送 Bedrock（JSON 中放多個 claims）。

**⛔ Grounding 工具可用性檢查（鐵律，不可跳過）：**

在 Phase 1b 開始時，必須先用 Bash 執行簡單測試驗證 grounding CLI 是否可用：

```bash
TEST_JSON='{"claims":["The sky is blue."],"sources":["The sky is blue during a clear day."]}'

# 1. 測試 Bedrock
echo "$TEST_JSON" | $PY $GS/bedrock-guardrails.py --cli 2>/dev/null

# 2. 若失敗，測試 MiniCheck
echo "$TEST_JSON" | $MINICHECK_PY $GS/minicheck.py --cli 2>/dev/null

# 3. 若也失敗，測試 Nemo
echo "$TEST_JSON" | $PY $GS/nemo-guardrails.py --cli 2>/dev/null
```

**判定規則：**
- 三者中至少一個必須成功返回有效 JSON（含 grounding_score 或 confidence）
- 若三者全部失敗 → **立即停止研究流程**，向使用者報告：
  ```
  ⛔ [GROUNDING-UNAVAILABLE] 所有 grounding 驗證工具均不可用：
  - Bedrock: {錯誤訊息}
  - MiniCheck: {錯誤訊息}
  - Nemo: {錯誤訊息}
  請修復後重新執行 /research。
  ```
- **禁止在沒有任何 grounding 工具可用的情況下繼續 Phase 1b**
- **禁止用「手動判斷」或「Claude 自行評估」替代 grounding 工具**

**Bedrock 只判定「claim 是否被提供的文本支持」，不判定「哪一方在現實中是真的」。禁止用 Bedrock 分數差直接裁決正反方真偽。**

---

# 1b-A：Grounding + 品質評估

## Step 1: Grounding 驗證

讀取 `workspace/search-results/` 目錄下的所有來源檔案。

對每個來源中的每個 QUOTE 和 NUMBER：
1. **Bedrock**：claim text + source text（超過 2000 tokens 時截取 quote 周邊 ±250 tokens）→ score
2. **含數字或直接引用** → 追加 Citation API

將結果寫入 `workspace/grounding-results/q{n}_grounding.md`。

---

## Step 2: 建立 Claim Ledger

寫入 `workspace/claim-ledger.md`：

```markdown
# Claim Ledger

| claim_id | subquestion | type | claim_text | source_ids | quote_ids | bedrock | citation | status |
|----------|-------------|------|------------|------------|-----------|---------|----------|--------|
| Q1-C1 | Q1 | numeric | "{完整 claim 文字}" | S003 | S003-N1 | 0.85 | precise | pending |
| Q1-C2 | Q1 | comparative | "{完整 claim 文字}" | S003,S005 | S003-Q1,S005-Q2 | 0.78 | N/A | pending |
```

**欄位說明：**
- `claim_id`：唯一識別碼，格式 Q{n}-C{m}
- `type`：numeric / comparative / causal / forecast / qualitative
- `claim_text`：**canonical text**，後續所有引用和核對都以此為準
- `quote_ids`：對應的逐字引用 ID（來自 search-results 檔案）
- `status`：pending / approved / rejected / needs_revision

**規則：claim_text 一旦寫入，後續 Phase 不得改寫。如需修正，建立新 claim_id。**

---

## Step 3: 4 維品質評估

對每個子問題評估：

| 維度 | Pass 標準 |
|------|----------|
| **Actionability** | 陳述具體、範圍清楚、限定詞正確。若來源本身不確定，保留不確定語氣不算 fail。 |
| **Freshness** | 核心數據符合 Phase 0 設定的 freshness SLA |
| **Plurality** | >= 2 個獨立來源（非同源轉述） |
| **Completeness** | 正反方 + 主要視角都有覆蓋 |

**評估結果：**
- 4/4 Pass → 該子問題的所有 pending claims 標記為 **approved** → 更新 coverage.chk
- 任一維度 Fail → 記錄失敗維度 → 進入 **1b-B**

---

## Step 4: 1b-A 快速通關判定

**如果所有子問題都 4/4 Pass：**
→ 跳過 1b-B，直接進入 Phase 2

**如果任一子問題有 Fail 維度：**
→ 僅對 Fail 的子問題進入 1b-B（已通過的子問題不需重做）

---

# 1b-B：Subagent 核對 + 辯證 + 補搜（條件觸發）

**只有 1b-A 未全過時才執行此段。**

## Step 5: Subagent 攻擊式核對

對每個 Fail 的子問題，spawn 一個 **Sonnet** subagent。多個子問題可平行。

**Subagent Prompt（填入後傳給 Agent tool，model: sonnet）：**

```
你是攻擊型事實核查員。你的任務是嘗試證明以下 claims 是錯的。

## 待核對 Claims
{從 claim-ledger.md 中提取該子問題所有 pending/needs_revision 的 claims，每個附 claim_id 和 claim_text}

## 搜尋結果檔案（用 Read 和 Glob 工具讀取）
目錄：{workspace 絕對路徑}/search-results/Q{n}/
讀取該目錄下所有 .md 檔案。
如果某檔案不存在，跳過，不得因缺檔中止。

## 核對規則（嚴格遵守）
對每個 claim：
1. 在搜尋結果中找到 QUOTE 或 NUMBER 原文，嘗試推翻 claim
2. 數字：執行逐字核對。15% ≠ 約15% ≠ 近15%。任何不一致 = NOT_SUPPORTED
3. 程度詞：原文「成長」但 claim 說「大幅成長」= PARTIAL
4. 語氣：原文「可能」但 claim 說「確定」= NOT_SUPPORTED
5. 跨來源拼接：如果 claim 需要兩個不同來源才能支持 = PARTIAL 並標記 COMPOSITE

若找不到逐字對應或明確支持的原文，必須判為 NOT_SUPPORTED。
禁止用「意思接近」代替支持。
禁止自己腦補可能的支持證據。

## 輸出格式（每個 claim 一段，嚴格遵守此格式）
CLAIM_ID: {claim_id}
VERDICT: SUPPORTED / PARTIAL / NOT_SUPPORTED
QUOTE_ID: {支持的 quote_id} 或 NONE
ISSUE: {如果非 SUPPORTED，具體說明問題類型和細節}
---
```

---

## Step 6: 處理 Subagent 結果

更新 claim-ledger.md 的 `status` 欄位：

| Subagent Verdict | 動作 | claim status |
|-----------------|------|-------------|
| SUPPORTED | 保留 | **approved** |
| PARTIAL | 修正 claim_text 使其與原文一致（建新 claim_id），或弱化語氣 | **needs_revision** → 修正後 approved |
| NOT_SUPPORTED | 觸發補搜 1 次。補搜後仍無支持 → 刪除 | **rejected** |

**Fail-Fast：** 同一 claim 補搜最多 2 次。第 3 次仍 NOT_SUPPORTED → 直接 rejected，不再迴圈。

---

## Step 7: 正反方辯證

1. 列出正方 approved claims + Bedrock 分數
2. 列出反方 approved claims + Bedrock 分數
3. 讀取 `prompts/ref-challenge-checklist.md`，從 22 項中**選出適用的項目**執行（不需全部通過，標記 N/A 的跳過）
4. 質疑成立 → 記入 `workspace/gap-log.md` 的「未解矛盾」區段，觸發補搜
5. **Perspective 來源中如有一手事實或實質反證** → 提升為 advocate/critic claim

---

## Step 8: 更新 Coverage Checklist

讀取 `workspace/coverage.chk`，用 Edit tool 更新每個項目：

| 更新規則 | 動作 |
|---------|------|
| 該 facet+role 有 ≥1 個 approved claim | `[ ]` → `[x]` + `evidence_found (S{ids})` |
| 搜過但無 approved claim（至少 2 次嘗試）| `[ ]` → `[x]` + `searched_2x_no_evidence` |
| 搜過但正在補搜 | 保持 `[ ]` + `in_progress` |

同時更新 `workspace/gap-log.md`（薄弱證據、缺失視角）。

---

## Step 9: 迭代決策

**觸發新迭代的條件：** 資料不足 / 矛盾未解 / coverage 有 required `[ ]` / Subagent NOT_SUPPORTED 未處理

**每輪結束時 Plan Reflection：**
1. 新發現了什麼？
2. 從搜尋結果中提取新術語 → 重寫下輪 query
3. 更新 gap-log.md
4. 重跑 4 維品質，delta = 0 → 飽和信號

**Beast Mode（搜尋預算 80% 時）：**
- 停止開新子問題和新 facet
- **但不能停止補足 required coverage 項目**
- 剩餘 20% 留給 Phase 2-3

**回到 Phase 1a 繼續搜尋 → 搜完回到 Phase 1b-A 驗證 → 循環直到停止條件**

---

## ⚠️ 結構化停止條件（全部成立才可停止）

```
□ 1. coverage.chk 中所有 required 項目皆已標為 [x]？
     → 有未標記的 required 項 = 禁止停止

□ 2. 每個 searched_2x_no_evidence 項目確實至少 2 次不同 query？
     → < 2 次 = 回到 Phase 1a 補搜

□ 3. 每個子問題 advocate 和 critic 各至少 1 個 approved claim？
     → 任一方 = 0 且非 searched_2x_no_evidence → 補搜

□ 4. 所有 high-risk claim（numeric/comparative/causal/forecast）
     都已完成 Grounding + URL check？
     → 列出 high-risk claims 和驗證狀態

□ 5. gap-log.md 的「未解矛盾」已全部處理或標記 [BLOCKER]？
     → 未處理的矛盾 > 0 = 處理完或標記 BLOCKER 才能停

□ 6. 所有 rejected claims 已處理（確認 rejected 或補搜後 approved）？

→ 全部 ✅ = 進入 Phase 2
→ 任一 ❌ = 回到 Phase 1a/Step 繼續
```

---

## 更新 execution-log.md

每輪結束時追加：

```
### 第 {R} 輪完成
🔍 搜尋：{N} 次（累計 {total}/{budget} = {pct}%）
📖 深讀：{N} 篇
🛡️ Grounding：{passed}✅ / {weak}⚠️ / {filtered}❌
📊 Claim Ledger：{approved} approved / {rejected} rejected / {pending} pending
📊 4 維品質：{pass}/4（Fail: {失敗維度}）
📊 1b-B 觸發？ {是/否}
📊 Coverage：{checked}/{total_required} required 項已完成
🔄 新 URL 新增率：{pct}% | approved claim 成長率：{pct}%
```
