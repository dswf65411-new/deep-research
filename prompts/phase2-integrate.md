# Phase 2: 整合 + 矛盾裁決

**輸入：** 讀取 workspace 中的：
- `workspace/claim-ledger.md`（**核心輸入：只用 status=approved 的 claims**）
- `workspace/source-registry.md`
- `workspace/search-results/`（原文引用用）
- `workspace/coverage.chk`（完整性檢查用）
- `workspace/gap-log.md`（未解矛盾用）

**輸出：** `workspace/report-sections/q{n}_section.md`

**本 Phase 必讀：** `prompts/source-criteria.md`、`prompts/ref-citation-embedding.md`

---

## 鐵律：只從 Approved Claims 生成

**讀取 claim-ledger.md，只使用 `status=approved` 的 claims。rejected 和 pending 的 claims 禁止出現在報告中。**

在開始整合前，先建立 **Approved Claims 清單**，整合時禁止引用此清單以外的任何事實或數字。

---

## Step 1: 匯入成果

讀取 workspace 檔案。為每個子問題整理 approved claims：
- 正方 claims + Bedrock 分數 + quote_ids
- 反方 claims + Bedrock 分數 + quote_ids
- 殘餘矛盾（正反方都有 approved claim 但結論相反）

---

## Step 2: 處理已定案的論點

| 狀態 | 動作 |
|------|------|
| 正反方達共識 + Bedrock >= 0.7 | 採納 |
| 正方有 approved claim，反方 coverage = searched_2x_no_evidence | 採納，但必須寫為「在本次已覆蓋的反方搜尋範圍內，未找到足以推翻此論點的高品質證據」（禁止全域式表述「反方未找到反駁」） |
| 反方有 approved claim，正方無有效回應 | 採納為風險/限制 |
| 兩方都有 approved claim 但結論相反 | → Step 3 矛盾裁決 |

---

## Step 3: 矛盾裁決（三段式，禁止 Bedrock 分差選邊）

**Bedrock 只判定「文本是否支持 claim」，不判定「哪一方在現實中是真的」。**

對矛盾論點：

**3a. 多維度比較（不只看 Bedrock）：**

| 比較維度 | 正方 | 反方 |
|---------|------|------|
| Source Tier | T{n} | T{n} |
| Independence（非同源） | {Y/N} | {Y/N} |
| Methodology Transparency | {Y/N} | {Y/N} |
| Freshness | {date} | {date} |
| Bedrock（參考，非決定性） | {score} | {score} |

**3b. 補搜裁決（如果上述比較無法分高下）：**
搜第三方 meta-analysis、原始數據 → 新來源也跑 Bedrock → 更新 claim-ledger

**3c. 無法裁決：**
2 輪補搜後仍無法分高下 → 保留 [CONFLICTING]，報告中呈現雙方觀點，記入 `gap-log.md` 的「未解矛盾」。（Fail-Fast：最多 2 輪，不做第 3 輪）

---

## Step 4: 信心等級 + 不確定性評分

**每個結論必須分配，無例外。**

| 等級 | 不確定性 | 條件 | 語氣規則 |
|------|---------|------|---------|
| 🟢 **HIGH** | < 0.1 | 2+ 獨立來源，≥1 個 T1-T2，Bedrock >= 0.7 | 可斷言 |
| 🟡 **MEDIUM** | 0.1-0.4 | 1-2 來源，T1-T4，Bedrock 0.5-0.7 | 審慎：「根據現有來源」 |
| 🟠 **CONFLICTING** | 0.4-0.7 | 正反方都有支持 | 呈現雙方 |
| 🔴 **LOW** | > 0.7 | 僅 T5-T6，或 Bedrock < 0.5 | 弱化：「有來源聲稱...但無法驗證」 |

**硬性規則：**
- 🔴 禁止斷言語氣
- 帶數字的結論必須 🟢 或 🟡，否則刪除數字或標記 [UNVERIFIED]
- **僅有 T4-T6 來源的子問題** → 最高只能評為 🟠，禁止輸出推薦句，禁止數字斷言

---

## Step 5: 來源深度評估

| 檢查項 | 動作 |
|--------|------|
| 原始性：Primary / Secondary / Tertiary | 標記 |
| 利益衝突 | 標記 [COI]，在結論中明示 |
| 同源：多來源轉述同一研究 | 只計 1 個獨立來源，**root-source 去重** |
| 時效：超過 freshness SLA | 降為「背景資訊」 |

---

## Step 6: 寫入報告段落

讀取 `prompts/ref-citation-embedding.md`，按其規則建立 verified-sources 約束。

為每個子問題寫入 `workspace/report-sections/q{n}_section.md`。**每完成一個就立即寫入。**

```markdown
# Q{n}: {子問題}
Status: FINAL
Based-On-Claims: Q{n}-C1, Q{n}-C2, ...

## 原文證據（正方）
> QUOTE[S{id}-Q{n}]: "{逐字引用}" — [{來源名稱}]({URL})
> NUMBER[S{id}-N{n}]: {數字} — Original: "{原句}" — [{來源名稱}]({URL})

## 原文證據（反方）
> QUOTE[S{id}-Q{n}]: "{逐字引用}" — [{來源名稱}]({URL})

## 數字對照表
| 報告數字 | 類型 | 原文原句 | 來源 | claim_id |
|---------|------|---------|------|----------|
| {數字} | ORIGINAL | "{原句}" | [{URL}] | Q{n}-C{m} |
| {換算數字} (orig: {原文數字+單位}) | NORMALIZED | "{原句}" | [{URL}] | Q{n}-C{m} |
| {計算結果} | DERIVED | 公式: {formula} | [{URL}] | Q{n}-C{m} |

## 分析與判斷 ← Claude 推論
{基於上述原文證據的分析}
{每句推論必須附 supporting claim_id}
{跨 claim 推導必須標記 [INFERENCE] 並列出所有 supporting claim_ids}

## 信心等級
- 等級：🟢/🟡/🟠/🔴
- 不確定性：{0.0-1.0}
- 依據：{來源數、層級、Bedrock 分數}
```

---

## Phase 2 完成 Checklist（逐項回答）

```
□ 1. 每個子問題都有 report-section 檔案（Status: FINAL）？
     → 逐一列出

□ 2. 每個事實陳述都有對應的 approved claim_id？
     → 無 claim_id 的事實陳述數：{N}（必須 = 0）

□ 3. 每個結論都有信心等級和不確定性評分？
     → 🟢:{n} 🟡:{n} 🟠:{n} 🔴:{n}

□ 4. 所有 🔴 結論已弱化語氣或移至「未解答」？
     → ✅/❌

□ 5. 所有非原文直接陳述是否標記 [INFERENCE] 並附 claim_ids？
     → 未標記的推論數：{N}（必須 = 0）

□ 6. root-source 去重已完成？所有 [COI] 在結論中明示？
     → ✅/❌

□ 7. gap-log.md 已更新（未解矛盾已記錄）？
     → ✅/❌

→ 全部 ✅ → 進入 Phase 3
```
