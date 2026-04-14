# Fallback Rules

When the pipeline gets stuck, do not spin idle; apply the following degradation steps. Record every degradation action in `workspace/gap-log.md`.

---

## Fetch

1. **WebFetch failure** -> three tiers: WebFetch -> Serper Scrape API -> mark `[UNREACHABLE]`.
   All three fail -> record in gap-log; do not write QUOTE/NUMBER based on search snippets alone.
2. **URL Health Check unavailable** -> try WebFetch to access; accessible = LIVE, inaccessible = [URL UNVERIFIED]

## Validation

3. **Bedrock API throttling/unavailable** -> wait 3 seconds and retry, up to 3 times. Continued failure -> switch to MiniCheck. MiniCheck-only claims:
   - Marked [FALLBACK_VERIFIED]
   - May not appear as numeric assertions in the final summary
   - MiniCheck also unavailable -> manual semantic comparison, annotated "tool validation unavailable"
4. **Bedrock weak + MiniCheck pass** -> still treated as WEAK, may not be upgraded
5. **Bedrock fail + MiniCheck pass** -> keep as disputed, do not auto-adopt
6. **Both fail** -> reject
7. **Citation API unavailable** -> skip citation-precision check, use Bedrock alone, annotate

## Sources

8. **Cannot find high-quality sources** -> fall back to T4-T6, with constraints:
   - The subquestion can at most be rated CONFLICTING or LOW
   - Recommendation sentences are forbidden
   - Numeric assertions are forbidden
   - The summary may only say "limited evidence suggests" or "community reports indicate"
9. **Multilingual results contradict** -> annotate "regional/language viewpoint difference", integrate and present

## Iteration

10. **Advocate and critic conclusions completely split** -> keep as unresolved contradiction, record under "Unresolved contradictions" in gap-log
11. **2 rounds of additional search still unresolved** -> mark [BLOCKER], record in gap-log, stop looping (Fail-Fast: at most 2 rounds)
12. **Search budget exhausted** -> produce the report with existing data, explain in the methodology section

## System

13. **Subagent spawn failure** -> main Agent reads the workspace files and compares them one by one
14. **Context too long and instructions forgotten** -> at the start of each Phase, re-Read the corresponding instruction file + claim-ledger + coverage.chk to keep state in sync
