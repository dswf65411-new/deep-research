# Phase 3: 報告生成 + 最終核對

**輸入：** 讀取 workspace 中的：
- `workspace/report-sections/*.md`（Status: FINAL 的段落）
- `workspace/claim-ledger.md`（approved claims）
- `workspace/source-registry.md`
- `workspace/search-results/`（最終核對用）
- `workspace/gap-log.md`（未解問題用）
- `workspace/phase0-plan.md`（元資料用）

**輸出：** `workspace/final-report.md` + `workspace/statement-ledger.md` + `workspace/execution-log.md`

**關鍵順序：先合併 → 先建 statement-ledger → 先核對 → 最後才寫摘要和圖表**

**本 Phase 必讀：** `prompts/ref-citation-embedding.md`

---

## Step 1: 合併報告段落

讀取所有 `workspace/report-sections/q{n}_section.md`（只讀 Status: FINAL 的），按子問題順序合併成報告主體。

**此時不寫摘要、不生成圖表。** 先完成核對。

---

## Step 2: 建立 Statement Ledger

將合併後的報告切成 statement 級別，寫入 `workspace/statement-ledger.md`：

```markdown
# Statement Ledger

| statement_id | section | text | claim_ids | type | verified |
|-------------|---------|------|-----------|------|----------|
| ST-1 | Q1-正方 | "{報告中的原句}" | Q1-C1 | fact | pending |
| ST-2 | Q1-分析 | "{推論句}" | Q1-C1,Q1-C2 | inference | pending |
| ST-3 | Q2-數字 | "{含數字的句}" | Q2-C3 | numeric | pending |
```

**type 分類：**
- `fact`：直接引用來源的事實
- `numeric`：含數字的陳述
- `inference`：跨 claim 推導（應已標記 [INFERENCE]）
- `opinion`：觀點表達（不需核對）

---

## Step 3: Subagent 最終核對

Spawn 一個 **Sonnet** subagent 做結構化核對。

**關鍵改進：subagent 核對 statement-ledger vs claim-ledger，不是自由掃全文。**

**Subagent Prompt（model: sonnet）：**

```
你是最終品質攻擊員。嘗試找出報告中的每一個錯誤。

## 待核對 Statements
讀取：{workspace 絕對路徑}/statement-ledger.md
只核對 type = fact / numeric / inference 的行（跳過 opinion）。

## 溯源帳本
讀取：{workspace 絕對路徑}/claim-ledger.md

## 搜尋結果
讀取：{workspace 絕對路徑}/search-results/ 目錄下所有檔案
（用 Glob 列出所有 .md 檔，逐一讀取。缺檔跳過。）

## 核對規則（嚴格遵守）
對每個 statement：

1. 溯源鏈完整？
   - statement 有 claim_id → claim_id 在 claim-ledger 中 status=approved → claim 有 quote_id → quote_id 在 search-results 中存在
   - 鏈中任一環斷裂 = BROKEN_CHAIN

2. 數字逐字核對？
   - statement 中的數字 vs claim_text 中的數字 vs QUOTE/NUMBER 原文
   - 任何不一致 = NUMBER_MISMATCH

3. 語氣一致？
   - 原文「可能」→ statement「確定」= TONE_MISMATCH

4. 組合型幻覺？
   - statement 引用 2+ claims 推出結論，但沒有任何單一來源說過這個結論 = COMPOSITE_HALLUCINATION

5. 過度推論？
   - [INFERENCE] 句的推導是否合理？是否超出 claim 範圍？ = OVER_INFERENCE

若找不到精確支持，必須報告。禁止用「意思接近」放行。
禁止跨來源拼出結論後判為 SUPPORTED。

## 輸出格式
STATEMENT_ID: {statement_id}
ISSUE: NONE / BROKEN_CHAIN / NUMBER_MISMATCH / TONE_MISMATCH / COMPOSITE_HALLUCINATION / OVER_INFERENCE / NO_SOURCE
DETAIL: {具體問題}
FIX: {修正建議}
---

最後輸出摘要：
TOTAL: {N} statements checked
PASS: {N}
FAIL: {N} (列出所有有問題的 statement_id)
```

---

## Step 4: 處理 Subagent 結果

更新 statement-ledger 的 `verified` 欄位：

| Issue | 動作 |
|-------|------|
| NONE | verified = pass |
| BROKEN_CHAIN | 補上缺失的 claim_id 或 quote_id，或刪除 statement |
| NUMBER_MISMATCH | 修正為原文數字 |
| TONE_MISMATCH | 弱化語氣 |
| COMPOSITE_HALLUCINATION | 標記 [INFERENCE] 或刪除 |
| OVER_INFERENCE | 加限定詞或移至「未解答問題」 |
| NO_SOURCE | 刪除或補搜（**補搜最多 2 次，Fail-Fast**） |

---

## Step 5: 引用元數據檢查（學術來源）

對引用**學術論文**的來源，讀取 `prompts/ref-citation-embedding.md` 的五分類學：

1. **標題存在性：** 完整標題精確搜尋
2. **元數據匹配：** 作者 + 標題 + 期刊 + 年份
3. **識別碼驗證：** DOI/arXiv ID 有效且內容一致

不通過 → [FABRICATED] → 從報告移除。

對**非學術來源**：確認 source-registry 中 `fetched_title` 與實際引用 title 一致。

---

## Step 6: Self-Critique

以「最挑剔的審稿人」角度：

1. 結論支撐度：每個結論有 claim_id 支持？有邏輯跳躍？
2. 反面充分性：一面倒？
3. 證據品質：過度依賴 T5-T6？
4. 完整性：遺漏面向？
5. 可操作性：結論夠具體？

嚴重 → 修正（補搜，最多 2 次 Fail-Fast）。中等 → 修正文字。輕微 → 修正格式。

---

## Step 7: 最終品質掃描

| 檢查項 | 標準 | 不通過 |
|--------|------|--------|
| 每個事實 statement 有 approved claim_id | 0 個斷鏈 | 刪除或補 |
| 每個 claim_id 有 quote_id/number_id | 溯源鏈完整 | 補上或刪除 |
| 正反方平衡 | 不是一面倒 | 補搜 |
| 數字有 ORIGINAL/NORMALIZED/DERIVED 標記 | 0 個無標記數字 | 補標記 |
| 無自我矛盾 | 邏輯一致 | 修正 |
| **CTran = 1.0** | 所有正反衝突如實呈現 | 加回遺漏的衝突 |

---

## Step 8: 現在才寫摘要和圖表

**所有核對通過後**，才生成：

**8a. 摘要（1-3 段）：**
- 只能從 approved claims（status=approved）重組
- 禁止從 report-section 的 prose 再自由摘要
- 每句摘要必須對應 claim_id

**8b. Mermaid 圖表（自動判斷）：**

| 主題特徵 | 圖表類型 |
|---------|---------|
| 流程/pipeline | `flowchart` |
| 時間演進 | `timeline` |
| 分類/結構 | `mindmap` |
| 比較 | markdown 比較表 |

不生成：Quick 模式、純 Q&A。

---

## Step 9: 組合最終報告

將以下內容組合寫入 `workspace/final-report.md`：

```markdown
# 研究報告：{主題}

**研究日期：** {YYYY-MM-DD}
**研究模式：** {模式}
**研究深度：** {深度}
**整體信心度：** {高/中/低} — {原因}
**搜尋統計：** {R} 輪，{N} 不重複 URL，{M} 篇深讀

## 摘要
{Step 8a 生成，每句附 claim_id}

## 視覺化概覽
{Step 8b 生成}

## 詳細分析
{合併 report-sections}

## 利害關係人視角
| 視角 | 觀點 | 來源 | Bedrock | claim_id |

## 正反方辯證記錄
| 論點 | 正方 | 反方 | Bedrock 正 | Bedrock 反 | 裁判 |

## 引用來源總表
| # | 來源 | 層級 | COI | 日期 | URL 狀態 | Bedrock |

## 未解答問題與知識缺口
{從 gap-log.md 匯入：缺失視角 + 薄弱證據 + 未解矛盾 + BLOCKER 項目}

## 研究方法論
```

---

## Step 10: 更新 execution-log 最終統計

```
## 最終統計

📊 報告品質：
  事實 statements：{n} | 通過核對：{n} | 修正：{n} | 刪除：{n}
  溯源鏈完整率：{n}/{total} = {pct}%
  CTran：{x}/{y} = {ratio}

📊 URL：
  ✅ LIVE：{n} | ⚠️ STALE：{n} | 🚫 HALLUCINATED：{n}

📊 不確定性分布：
  🟢:{n} | 🟡:{n} | 🟠:{n} | 🔴:{n}

📊 Claim Ledger：
  Total：{n} | Approved：{n} | Rejected：{n}

📊 搜尋總計：
  搜尋次數：{n}/{budget} | 不重複 URL：{n} | 深讀：{n} 篇 | 迭代：{R} 輪

📊 Coverage Matrix：
  evidence_found：{n}/{total_required}
  searched_no_evidence：{n}
```

---

## Step 11: 呈現給使用者

輸出最終報告，提供後續選項：

```
📌 後續選項：
1. 針對某子問題深入研究
2. 更新特定數據
3. 匯出 PDF（/markdown-to-pdf）
4. 針對 🟠/🔴 論點額外驗證
5. 用測試驗證報告結論
```

---

## Phase 3 完成 Checklist

```
□ 1. Statement-ledger 已建立？ → {N} statements
□ 2. Subagent 最終核對完成？ → PASS:{n} FAIL:{n}
□ 3. 所有 FAIL statements 已處理？ → ✅
□ 4. 學術引用元數據通過？ → {N} 篇驗證
□ 5. Self-Critique 完成？ → 嚴重:{n} 中等:{n} 輕微:{n}
□ 6. 最終品質掃描通過？ → ✅
□ 7. CTran = 1.0？ → {ratio}
□ 8. 摘要只從 approved claims 生成？ → ✅
□ 9. final-report.md 已寫入？ → ✅
□ 10. execution-log.md 已更新？ → ✅
□ 11. gap-log.md 的所有項目已反映在「未解答問題」段落？ → ✅
→ 全部 ✅ → 研究完成
```
