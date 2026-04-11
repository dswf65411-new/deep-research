#!/usr/bin/env python3
"""
MiniCheck MCP Server — Efficient Claim-Level Fact-Checking
Uses MiniCheck-FT5 (770M params) to verify whether claims are supported by source documents.

Performance: 74.7% balanced accuracy (comparable to GPT-4's 75.3%) at 400× lower cost.
Based on: arxiv 2404.10774 — "Efficient Fact-Checking of LLMs on Grounding Documents"

Requires: pip3 install torch transformers sentencepiece
Model auto-downloads on first use (~1.5GB): Bespoke-MiniCheck-7B or lytang/MiniCheck-Flan-T5-Large
"""

import asyncio
import json
import sys
import os
from typing import Any

# Lazy-load heavy imports
_model = None
_tokenizer = None
_device = None


def _load_model():
    """Load MiniCheck model on first call. Uses MPS on Apple Silicon."""
    global _model, _tokenizer, _device

    if _model is not None:
        return

    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    model_name = "lytang/MiniCheck-Flan-T5-Large"

    sys.stderr.write(f"Loading MiniCheck model: {model_name}...\n")
    sys.stderr.flush()

    _tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Force CPU — MPS has compatibility issues with transformers generate() on macOS
    # CPU performance is acceptable: ~0.5-0.7s per claim after model load
    _device = "cpu"

    _model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(_device)
    _model.eval()

    sys.stderr.write(f"MiniCheck loaded on {_device}\n")
    sys.stderr.flush()


def check_claim(claim: str, source: str) -> dict:
    """
    Check if a claim is supported by the source document.
    Returns: label (1=supported, 0=not supported) and confidence score.
    """
    import torch

    _load_model()

    # MiniCheck input format: "premise: {source} hypothesis: {claim}"
    input_text = f"premise: {source} hypothesis: {claim}"
    inputs = _tokenizer(input_text, return_tensors="pt", max_length=2048, truncation=True).to(_device)

    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=10,
            output_scores=True,
            return_dict_in_generate=True,
        )

    # Decode the generated token
    generated_text = _tokenizer.decode(outputs.sequences[0], skip_special_tokens=True).strip().lower()

    # MiniCheck outputs "1" for supported, "0" for not supported
    supported = generated_text in ("1", "true", "yes", "entailment")

    # Get confidence from logits
    if outputs.scores:
        logits = outputs.scores[0][0]
        probs = torch.softmax(logits, dim=-1)
        confidence = float(probs.max())
    else:
        confidence = 1.0 if supported else 0.0

    return {
        "claim": claim,
        "supported": supported,
        "label": 1 if supported else 0,
        "confidence": round(confidence, 4),
        "raw_output": generated_text,
    }


def batch_check_claims(claims: list[str], sources: list[str]) -> dict:
    """Check multiple claims against source documents."""
    combined_source = "\n\n".join(sources)
    results = []

    for claim in claims:
        try:
            result = check_claim(claim, combined_source)
            results.append(result)
        except Exception as e:
            results.append({
                "claim": claim,
                "supported": False,
                "label": 0,
                "confidence": 0.0,
                "error": str(e),
            })

    supported_count = sum(1 for r in results if r.get("supported", False))
    total = len(results)
    avg_confidence = round(
        sum(r.get("confidence", 0) for r in results) / total, 4
    ) if total > 0 else 0

    return {
        "summary": {
            "total_claims": total,
            "supported": supported_count,
            "not_supported": total - supported_count,
            "support_rate": round(supported_count / total, 3) if total > 0 else 0,
            "average_confidence": avg_confidence,
            "model": "lytang/MiniCheck-Flan-T5-Large",
            "device": _device or "not_loaded",
        },
        "results": results,
    }


# ── MCP Server ──

TOOLS = [
    {
        "name": "minicheck_verify",
        "description": (
            "Verify whether AI-generated claims are supported by source documents "
            "using MiniCheck-FT5 (770M params, local execution). "
            "Accuracy comparable to GPT-4 (74.7% vs 75.3%) at 400× lower cost. "
            "Returns per-claim support label and confidence score. "
            "Runs locally on Apple Silicon (MPS) — no API costs. "
            "First call loads model (~30s), subsequent calls are fast (~1s per claim)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "claims": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of claims to verify against source documents"
                },
                "sources": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of source document texts to check claims against"
                }
            },
            "required": ["claims", "sources"]
        }
    }
]


async def handle_request(msg: dict):
    method = msg.get("method", "")
    id = msg.get("id")
    params = msg.get("params", {})

    if method == "initialize":
        write_message(make_response(id, {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "minicheck", "version": "1.0.0"}
        }))
    elif method == "notifications/initialized":
        pass
    elif method == "tools/list":
        write_message(make_response(id, {"tools": TOOLS}))
    elif method == "tools/call":
        tool_name = params.get("name", "")
        args = params.get("arguments", {})

        if tool_name == "minicheck_verify":
            try:
                result = batch_check_claims(
                    claims=args["claims"],
                    sources=args["sources"]
                )
                write_message(make_response(id, {
                    "content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False, indent=2)}]
                }))
            except ImportError as e:
                write_message(make_response(id, {
                    "content": [{"type": "text", "text": json.dumps({
                        "error": f"Missing dependency: {e}. Run: pip3 install torch transformers sentencepiece"
                    })}],
                    "isError": True
                }))
            except Exception as e:
                write_message(make_response(id, {
                    "content": [{"type": "text", "text": f"Error: {str(e)}"}],
                    "isError": True
                }))
        else:
            write_message(make_error(id, -32601, f"Unknown tool: {tool_name}"))
    elif method == "ping":
        write_message(make_response(id, {}))
    else:
        if id is not None:
            write_message(make_error(id, -32601, f"Unknown method: {method}"))


async def read_message() -> dict:
    line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
    if not line:
        raise EOFError()
    return json.loads(line.strip())


def write_message(msg: dict):
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def make_response(id: Any, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": id, "result": result}


def make_error(id: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}}


async def main():
    while True:
        try:
            msg = await read_message()
            await handle_request(msg)
        except EOFError:
            break
        except json.JSONDecodeError:
            continue
        except Exception as e:
            sys.stderr.write(f"Error: {e}\n")
            sys.stderr.flush()


if __name__ == "__main__":
    if "--cli" in sys.argv:
        # CLI mode: read JSON from stdin, output JSON to stdout
        data = json.loads(sys.stdin.read())
        result = batch_check_claims(
            claims=data["claims"],
            sources=data["sources"],
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        asyncio.run(main())
