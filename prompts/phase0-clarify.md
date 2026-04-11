# Phase 0: 澄清 + 研究規劃

**輸入：** 使用者的研究主題
**輸出：** `workspace/phase0-plan.md` + `workspace/coverage.chk` + `workspace/gap-log.md`
**完成條件：** 使用者確認研究計畫

---

## Step 1: 澄清提問

用 AskUserQuestion 確認。意圖明確就少問，模糊就多問。每個問題說明「為什麼需要知道」。

1. **研究目的：** 做決策？寫報告？學習？解決問題？
2. **期望產出：** 比較表？推薦結論？客觀呈現？
3. **範圍邊界：** 包含/排除什麼？時間範圍？
4. **背景知識：** 已知什麼？已排除什麼？
5. **特定偏好：** 技術棧、地區、語言、必須涵蓋的來源？

---

## Step 2: 題型分流 + 研究模式

**2a. 題型判斷：**

| 題型 | 判斷標準 | Phase 0 要求 |
|------|---------|-------------|
| **單點查證** | 子問題 < 3，無相依性 | 簡化：只需目的、範圍、來源優先級、幻覺錨點。不需完整 DAG。 |
| **比較/決策** | 有明確選項需比較 | 完整流程：DAG + Adversarial |
| **趨勢/政策** | 涉及時間軸或多方利害關係人 | 完整流程 + Multi-Stakeholder/Temporal |

**2b. 研究模式選擇（預設 Adversarial，可組合 1-2 種）：**

| 研究目的 | 模式 | Phase 1 策略 |
|---------|------|-------------|
| 做決策 | **Adversarial** | 正反方辯證 |
| 趨勢分析 | **Temporal** | 歷史→現狀→預測 |
| 探索新領域 | **Funnel** | 廣掃→深鑽→合成 |
| 政策/影響評估 | **Multi-Stakeholder** | N 方各自分析→合成 |
| 比較評測 | **Adversarial + Matrix** | 辯證 + 比較矩陣 |

---

## Step 3: Query Enrichment

**3a. Specificity Maximization：**
- 已指定的維度 → 列出
- 未指定但重要的維度 → 列出 + 標注處理方式

**3b. Source Prioritization：**
官方文件 > 原始論文 > 行業報告 > 技術部落格 > 社群討論

**3c. PICO Framing：**
- Population/Problem：研究對象/問題
- Intervention：評估的方案/技術
- Comparison：比較基準/替代方案
- Outcome：期望的結果/指標

**3d. Freshness SLA（按主題設定，非一刀切）：**

| Claim 類型 | 預設時效 | 可調整 |
|-----------|---------|--------|
| 數字型（價格、性能、市佔率） | 12 個月 | 使用者可放寬 |
| 政策/法規型 | 24 個月 | — |
| 背景/理論型 | 36 個月 | — |
| 歷史一手來源 | **豁免** | — |

**3e. Anti-Hallucination Anchors：**
- 容易幻覺的地方（數字/因果/趨勢）
- 必須從官方來源驗證的項目

---

## Step 4: Perspective Discovery

用 1-2 次 WebSearch 搜尋 `{主題} perspectives` 或 `{主題} stakeholders impact`。

| 視角 | 代表群體 | 核心關注 | 專屬搜尋角度 |
|------|---------|---------|------------|
| （至少 3 個） | | | |

**重要：** perspective 搜尋結果預設為補充觀點；但若其中包含可驗證的一手事實或實質反證，必須提升為 advocate/critic claim 參與辯證。

---

## Step 5: 子問題 DAG

（單點查證題可跳過，直接列 1-2 個子問題）

```
[獨立] Q1: {描述} ──┐
[獨立] Q2: {描述} ──┼── [依賴Q1+Q2] Q4: {描述}
[獨立] Q3: {描述} ──┘
```

每個子問題定義 **facets**（搜尋面向）：

```
Q1:
  facets: [benchmark, adoption, cost, risks]
  must_cover_roles: [advocate, critic]
```

---

## Step 6: 搜尋策略

為每個子問題設計，但**第一輪不展開所有 query**：

```
### Q{n}: {子問題}

第一輪（最小集）：
- advocate: 1 個 query family（EN + ZH）
- critic: 1 個 query family（EN + ZH）
- perspective: 0-1 個（如有明確視角）
- academic: 0-1 個（僅學術/技術主題）

後續輪次 query 必須由以下觸發：
- coverage.chk 中未標記 [x] 的面向
- challenge checklist 發現的缺口
- Query Rewriting（從搜尋結果中提取新術語）
禁止第一輪就展開所有 query。
```

搜尋廣度覆蓋（確保涵蓋）：

| 來源類型 | 搜尋方式 | 優先級 |
|---------|---------|--------|
| 學術論文 | "survey"/"paper" + arxiv/scholar | 高 |
| 官方文件 | 官方 docs/changelog/pricing | 高 |
| 業界報告 | "benchmark"/"report" | 高 |
| 開發者社群 | reddit/HN/GitHub | 中 |
| 一般社群 | 部落格/Medium/知乎 | 中 |
| 新聞媒體 | TechCrunch/The Verge 等 | 依主題 |

---

## Step 7: 搜尋預算分配

```
總預算：{30 / 60 / 150} 次

| 類別 | 比例 | 分配給 |
|------|------|--------|
| 核心問題 | 40% | 高不確定性子問題 |
| 支撐問題 | 20% | 背景性子問題 |
| 視角補充 | 10% | 各視角 |
| 迭代儲備 | 20% | Gap Queue + Query Rewriting |
| Phase 2-3 | 10% | 矛盾裁決 + 報告驗證 |
```

---

## Step 8: 建立 Coverage Checklist + Gap Log

**這是防止「搜尋停太早」的核心機制。用簡易 checklist 取代複雜 table，降低格式錯誤。**

寫入 `workspace/coverage.chk`：

```
# Coverage Checklist

## Q1: {子問題}
- [ ] advocate:benchmark — not_started
- [ ] critic:benchmark — not_started
- [ ] advocate:adoption — not_started
- [ ] critic:risks — not_started
- [ ] perspective:regulator — not_started (optional)

## Q2: {子問題}
- [ ] advocate:performance — not_started
- [ ] critic:performance — not_started
```

同時初始化 `workspace/gap-log.md`：

```markdown
# Gap Log

## 缺失視角
（Phase 1 搜尋過程中發現但尚未覆蓋的立場）

## 薄弱證據
（只有單一來源的 claim）

## 未解矛盾
（正反方都有 approved claim 但結論相反）
```

**規則：**
- 無 `(optional)` 標記的項目 = required，必須在 Phase 1 結束前標為 `[x]`（evidence_found）或記錄 `searched_2x_no_evidence`
- `(optional)` 項目盡力搜尋，但不阻止進入 Phase 2
- **禁止刪除 checklist 項目來提高覆蓋率**
- 更新方式：用 Edit tool 將 `[ ]` 改為 `[x]` 並更新狀態描述

---

## Step 9: 寫入 workspace + 呈現

將所有結果寫入 `workspace/phase0-plan.md`，格式：

```markdown
# 研究計畫

## 結構化 Header
- topic: {主題}
- mode: {Adversarial / Temporal / Funnel / Multi-Stakeholder / 組合}
- depth: {Quick / Standard / Deep}
- budget: {30 / 60 / 150}
- freshness_sla:
  - numeric: {N} 個月
  - policy: {N} 個月
  - background: {N} 個月
  - historical_exempt: true/false
- subquestions: {N} 個
- perspectives: {N} 個
- total_coverage_units: {N} 個（required: {M}）

## Query Enrichment
{PICO + 來源優先級 + 防幻覺錨點}

## 利害關係人視角
{視角清單 + 各自關注和搜尋角度}

## 子問題 DAG
{子問題 + facets + 依賴 + 執行順序}

## 搜尋策略
{第一輪最小 query 集 + 後續觸發規則}

## 預算分配
{分配表}

## 幻覺高風險區域
{哪些論點需特別驗證：數字型/因果型/趨勢型/比較型}

## 納入/排除標準
- 納入：{語言、時間、地域、來源類型}
- 排除：{排除項}

請確認此研究計畫，或提出修改。
```

同時確認 `workspace/coverage.chk` 和 `workspace/gap-log.md` 已寫入。

等使用者確認後進入 Phase 1a。

---

## Gate Check（必須全過才能離開 Phase 0）

```
□ 1. phase0-plan.md 已寫入？ → ✅
□ 2. coverage.chk 已寫入？ → {N} 個項目（required: {M}）
□ 3. gap-log.md 已初始化？ → ✅
□ 4. 研究模式已選定？ → {模式}
□ 5. 子問題已拆解（含 facets 和依賴）？ → {N} 個子問題
□ 6. 使用者已確認？ → ✅
→ 全部 ✅ → 進入 Phase 1a
```
