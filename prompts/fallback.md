# Fallback 規則

當流程卡住時，不空轉，按以下降級處理。所有降級動作都記入 `workspace/gap-log.md`。

---

## 抓取類

1. **WebFetch 失敗** → 三階梯：WebFetch → Serper Scrape API → 標記 `[UNREACHABLE]`。
   三階梯全敗 → 記入 gap-log，不得僅依賴搜尋摘要寫入 QUOTE/NUMBER。
2. **URL Health Check 不可用** → WebFetch 嘗試存取，能存取 = LIVE，不能 = [URL UNVERIFIED]

## 驗證類

3. **Bedrock API throttling/不可用** → 等待 3 秒重試，最多 3 次。持續失敗 → 改用 MiniCheck。MiniCheck-only 的 claim：
   - 標記 [FALLBACK_VERIFIED]
   - 不得進入最終摘要的數字型斷言
   - MiniCheck 也不可用 → 人工語義比對，標註「工具驗證不可用」
4. **Bedrock weak + MiniCheck pass** → 仍視為 WEAK，不得升級
5. **Bedrock fail + MiniCheck pass** → 保留爭議，不自動採納
6. **兩者皆 fail** → reject
7. **Citation API 不可用** → 跳過引用精確度檢查，僅用 Bedrock，標註

## 來源類

8. **搜尋不到高品質來源** → 降級用 T4-T6，但同步限制：
   - 該子問題最高只能評為 🟠 CONFLICTING 或 🔴 LOW
   - 禁止輸出推薦句
   - 禁止數字型斷言
   - 摘要只能寫「有限證據顯示」或「社群回報指出」
9. **多語言結果矛盾** → 標註「地域/語言觀點差異」，整合呈現

## 迭代類

10. **正反方結論完全分裂** → 保留未解矛盾，記入 gap-log「未解矛盾」
11. **補搜 2 輪仍無法解決** → 標記 [BLOCKER]，記入 gap-log，停止迴圈（Fail-Fast：最多 2 輪）
12. **搜尋預算耗盡** → 用現有資料產出報告，在方法論中說明

## 系統類

13. **Subagent spawn 失敗** → 主 Agent 自行讀 workspace 檔案逐一比對
14. **Context 過長導致指令遺忘** → 每個 Phase 開始時重新 Read 對應指令檔 + claim-ledger + coverage.chk，確保狀態同步
