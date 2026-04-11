# 本地驗證工具 CLI 參考

所有驗證工具為 CLI 模式，腳本位於專案內 `grounding_scripts/`。統一透過 Bash tool 呼叫，輸入 JSON 至 stdin，輸出 JSON 至 stdout。

## 路徑設定
```
PY=<project>/.venv/bin/python3
GS=<project>/grounding_scripts
MINICHECK_PY=<project>/.minicheck-venv/bin/python3
```

---

## 1. Bedrock Grounding Check（主要驗證）

```bash
echo '{"claims":["claim1","claim2"],"sources":["source text"]}' | $PY $GS/bedrock-guardrails.py --cli
```

可選參數：`guardrail_id`（預設 981o7pz3ze8q）、`threshold`（預設 0.7）、`region`（預設 us-east-1）

輸出：`summary.grounding_rate` + 每個 claim 的 `grounding_score` 和 `verdict`（GROUNDED / NOT_GROUNDED）

---

## 2. MiniCheck（備用驗證）

```bash
echo '{"claims":["claim1"],"sources":["source text"]}' | $MINICHECK_PY $GS/minicheck.py --cli
```

注意：首次呼叫需載入模型 ~30 秒，後續 ~1 秒/claim。

輸出：`summary.support_rate` + 每個 claim 的 `confidence` 和 `supported`（true/false）

---

## 3. NeMo Grounding Check（第三備用）

```bash
echo '{"claims":["claim1"],"sources":["source text"],"threshold":0.7}' | $PY $GS/nemo-guardrails.py --cli
```

輸出：`summary.grounding_rate` + 每個 claim 的 `grounding_score` 和 `verdict`

---

## 4. URL Health Check

```bash
echo '{"urls":["https://example.com"],"timeout":15}' | $PY $GS/urlhealth.py --cli
```

輸出：每個 URL 的 `status`（LIVE / STALE / LIKELY_HALLUCINATED / UNKNOWN）

---

## Grounding 工具可用性檢查

Phase 1b 開始前必須執行：

```bash
# 測試 Bedrock
echo '{"claims":["The sky is blue."],"sources":["The sky is blue during a clear day."]}' | $PY $GS/bedrock-guardrails.py --cli 2>/dev/null

# 若失敗，測試 MiniCheck
echo '{"claims":["The sky is blue."],"sources":["The sky is blue during a clear day."]}' | $MINICHECK_PY $GS/minicheck.py --cli 2>/dev/null

# 若也失敗，測試 Nemo
echo '{"claims":["The sky is blue."],"sources":["The sky is blue during a clear day."]}' | $PY $GS/nemo-guardrails.py --cli 2>/dev/null
```

三者全部失敗 → 停止研究，報錯 `[GROUNDING-UNAVAILABLE]`。
