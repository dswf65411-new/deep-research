# Focused Task Execution（共用規則，引用於各 phase prompt）

**這不是可選建議，是必須遵守的執行契約。**

當你被交付一個涉及多步驟的任務時，**不要一次想完所有事**。請按以下流程：

## Step 1 — 列出 Internal Task List（先思考，後執行）

在回應最前面，先輸出一個結構化任務清單，列出本次要做的**所有步驟與子步驟**：

```
[TASK LIST]
T1. <第一步>
T2. <第二步>
  T2.1 <T2 的子步驟>
  T2.2 <T2 的子步驟>
T3. <第三步>
...
```

規則：
- 子步驟用 `  ` 縮排（兩空格）
- 每個 task 一行，動詞開頭，簡短具體（≤15 字）
- 若看到任務需要產生多個獨立輸出（例如多個 query、多個 claim、多個 section），**每個輸出一個獨立 task**
- 禁止塞話（不要在 task 名稱裡寫理由/限制）
- 任務總數超過 10 → 合併成類別
- 任務只有 1 個 → 可省略整個 [TASK LIST]（表示任務夠原子）

## Step 2 — 逐項執行（聚焦當下）

依序處理每個 task。在處理當下 task 時：

```
[WORKING: T{n}]
（僅針對 T{n} 的具體工作 —— 收集、分析、生成輸出）
[DONE: T{n}]
```

禁止：
- 跳躍執行（做 T1 中途又回頭改 T3）
- 並行思考（一次帶入多個 task 的 context）
- 省略 [WORKING]/[DONE] 標記（下游 parser 和人工 review 靠這些對位）

## Step 3 — 最終輸出（符合任務要求的格式）

所有 task 完成後，輸出任務真正要求的格式（JSON / markdown / plain text，依 prompt 指定）。

**不要把 [TASK LIST] / [WORKING] / [DONE] 放進最終交付的資料結構裡**（例如 JSON output）。那些只是思考的 scaffolding，最終產出是乾淨的結果。若 prompt 要 JSON，最終那份 JSON 就是答案，前面 scaffold 留在 plain text 區。

## 為什麼這樣做

LLM 一次處理多件事容易：
- 忘記某件事（特別是列表後半）
- 中途切換 context 污染輸出
- 產生合理但不完整的答案（看起來對，仔細看缺東西）

強制「先列清單、逐項處理、最終輸出」把每個 token 的決策空間壓到最小 —— 當下只需要想一件事。

## 什麼情況可以省略

- 任務本身是單一判斷（e.g.「這個 claim 是否 on-topic」yes/no）
- 任務只需輸出一個 short string（e.g. 生成單一 query）
- 任務是純翻譯或格式轉換

上述情況直接回答即可，不需 scaffold。
