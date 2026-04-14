# Pre-generation Citation Embedding (citation-constraint template)

Before extracting claims from the source text, embed the validated sources into the prompt as a hard constraint so that citations may only come from the pre-embedded sources.

## Template

```xml
<verified-sources>
  <source id="S1" url="{url}" status="LIVE" tier="{T1-T6}" fetch_date="{YYYY-MM-DD}">
    <title>{original title}</title>
    <authors>{authors (if any)}</authors>
    <excerpt>{key paragraph copied verbatim from the original, not a paraphrased summary}</excerpt>
  </source>
  <source id="S2" ...>...</source>
</verified-sources>

[Hard constraints — violating any is treated as hallucination]:
1. Every claim must carry a source id tag (e.g. [S1][S3])
2. Citing sources outside <verified-sources> is forbidden
3. Deriving conclusions that the original text does not state is forbidden (mark [INFERENCE] when derivation is needed)
4. Rewriting the numbers in the original text is forbidden (must match exactly)
5. When information is insufficient, answer "the available validated sources do not cover this facet"
```

## Metadata Requirements

Every source must include url, status, tier, fetch_date, title. Missing any field -> [INCOMPLETE METADATA], may not be the sole source.

## Citation Quantity Cap

Each subquestion conclusion may cite at most **5 URLs**. Selection criteria: Bedrock >= 0.8 > T1-T2 > cross-engine hit. Better to cite 3 strong sources than 15 uneven ones.

**Basis**: arxiv 2604.03173 — models that produce 4.3x more citations have 2x higher error rates. Perplexity embeds citation markers before generation to constrain generation behavior.

## Five-Way Taxonomy of Citation Hallucinations (for validation)

| Type | Description | Share | Check method |
|------|------|------|---------|
| **TF** Total fabrication | Does not exist at all | 66% | URL liveness + academic search |
| **PAC** Partial attribute corruption | Real author paired with wrong paper | 27% | Multi-attribute cross-check |
| **IH** Identifier hijacking | DOI is valid but content does not match | 4% | Open and confirm consistency |
| **SH** Semantic hallucination | Sounds plausible but does not exist | 1% | Exact search of full title |
| **PH** Placeholder | XXXX, Firstname, etc. | 2% | Regex scan |

Every citation must at least complete the TF + PAC checks (covering 93%). Academic papers get an additional IH check.
