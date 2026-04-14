# Local Validation Tools CLI Reference

All validation tools run in CLI mode; scripts live under `grounding_scripts/` in the project. They are called uniformly via the Bash tool, with JSON piped into stdin and JSON emitted to stdout.

## Path Setup
```
PY=<project>/.venv/bin/python3
GS=<project>/grounding_scripts
MINICHECK_PY=<project>/.minicheck-venv/bin/python3
```

---

## 1. Bedrock Grounding Check (primary validator)

```bash
echo '{"claims":["claim1","claim2"],"sources":["source text"]}' | $PY $GS/bedrock-guardrails.py --cli
```

Optional arguments: `guardrail_id` (default 981o7pz3ze8q), `threshold` (default 0.7), `region` (default us-east-1)

Output: `summary.grounding_rate` + per-claim `grounding_score` and `verdict` (GROUNDED / NOT_GROUNDED)

---

## 2. MiniCheck (fallback validator)

```bash
echo '{"claims":["claim1"],"sources":["source text"]}' | $MINICHECK_PY $GS/minicheck.py --cli
```

Note: the first call loads the model in ~30 seconds; subsequent calls are ~1 second per claim.

Output: `summary.support_rate` + per-claim `confidence` and `supported` (true/false)

---

## 3. NeMo Grounding Check (third fallback)

```bash
echo '{"claims":["claim1"],"sources":["source text"],"threshold":0.7}' | $PY $GS/nemo-guardrails.py --cli
```

Output: `summary.grounding_rate` + per-claim `grounding_score` and `verdict`

---

## 4. URL Health Check

```bash
echo '{"urls":["https://example.com"],"timeout":15}' | $PY $GS/urlhealth.py --cli
```

Output: per-URL `status` (LIVE / STALE / LIKELY_HALLUCINATED / UNKNOWN)

---

## Grounding Tool Availability Check

Must be run before Phase 1b:

```bash
# Test Bedrock
echo '{"claims":["The sky is blue."],"sources":["The sky is blue during a clear day."]}' | $PY $GS/bedrock-guardrails.py --cli 2>/dev/null

# If it fails, test MiniCheck
echo '{"claims":["The sky is blue."],"sources":["The sky is blue during a clear day."]}' | $MINICHECK_PY $GS/minicheck.py --cli 2>/dev/null

# If that also fails, test Nemo
echo '{"claims":["The sky is blue."],"sources":["The sky is blue during a clear day."]}' | $PY $GS/nemo-guardrails.py --cli 2>/dev/null
```

All three fail -> stop research, raise `[GROUNDING-UNAVAILABLE]`.
