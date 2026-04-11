# Pre-generation Citation Embedding（引用約束模板）

在從原文提取論點之前，將已驗證來源嵌入 prompt，硬性約束只能從預嵌入來源引用。

## 模板

```xml
<verified-sources>
  <source id="S1" url="{url}" status="LIVE" tier="{T1-T6}" fetch_date="{YYYY-MM-DD}">
    <title>{原文標題}</title>
    <authors>{作者（如有）}</authors>
    <excerpt>{從原文複製的關鍵段落，不是摘要改寫}</excerpt>
  </source>
  <source id="S2" ...>...</source>
</verified-sources>

【硬性約束 — 違反任何一條即視為幻覺】：
1. 每個論點必須標注 source id（如 [S1][S3]）
2. 禁止引用 <verified-sources> 之外的來源
3. 禁止推導原文都沒說的結論（需推導則標記 [INFERENCE]）
4. 禁止改寫原文數字（必須完全一致）
5. 資料不足時回答「現有已驗證來源未涵蓋此面向」
```

## 元數據要求

每個 source 必須含 url、status、tier、fetch_date、title。缺欄位 → [INCOMPLETE METADATA]，不可為唯一來源。

## 引用數量上限

每個子問題結論最多引用 **5 個 URL**。選擇標準：Bedrock >= 0.8 > T1-T2 > 跨引擎命中。寧引 3 強來源，不引 15 個參差不齊的。

**依據**：arxiv 2604.03173 — 產出 4.3× 更多引用的模型，錯誤率高 2×。Perplexity 在生成前嵌入引用標記約束生成行為。

## 引用幻覺五分類學（驗證用）

| 類型 | 說明 | 佔比 | 檢查方式 |
|------|------|------|---------|
| **TF** 完全捏造 | 全部不存在 | 66% | URL 活性 + 學術搜尋 |
| **PAC** 部分篡改 | 真實作者配錯論文 | 27% | 多屬性交叉比對 |
| **IH** 識別碼劫持 | DOI 有效但內容不符 | 4% | 點開確認一致 |
| **SH** 語義幻覺 | 聽起來合理但不存在 | 1% | 完整標題精確搜尋 |
| **PH** 佔位符 | XXXX、Firstname 等 | 2% | 正則掃描 |

每個引用至少完成 TF + PAC 檢查（覆蓋 93%）。學術論文追加 IH 檢查。
