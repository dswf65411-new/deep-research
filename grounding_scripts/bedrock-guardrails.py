#!/usr/bin/env python3
"""
Amazon Bedrock Guardrails — Contextual Grounding Check (CLI mode)
Uses AWS Bedrock's managed grounding service to verify AI claims against sources.

Setup steps:
1. Create AWS account and enable Bedrock in your region
2. Run: aws configure (set access key, secret, region)
"""

import asyncio
import json
import sys
import os
from typing import Any


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


# ── Bedrock Grounding Check ──

def check_grounding_bedrock(
    claims: list[str],
    sources: list[str],
    guardrail_id: str,
    guardrail_version: str = "DRAFT",
    region: str = "us-east-1",
    threshold: float = 0.7
) -> dict:
    """
    Use Amazon Bedrock Guardrails Contextual Grounding Check to verify claims.
    """
    import boto3

    client = boto3.client("bedrock-runtime", region_name=region)

    # Combine all sources into a single grounding source text
    combined_source = "\n\n".join(sources)

    results = []
    for claim in claims:
        try:
            # Bedrock ApplyGuardrail API requires 3 content elements:
            # 1. grounding_source: the reference documents
            # 2. query: the original question/context
            # 3. No qualifier: the output text to verify against sources
            response = client.apply_guardrail(
                guardrailIdentifier=guardrail_id,
                guardrailVersion=guardrail_version,
                source="OUTPUT",
                content=[
                    {
                        "text": {
                            "text": combined_source,
                            "qualifiers": ["grounding_source"]
                        }
                    },
                    {
                        "text": {
                            "text": "Is the following claim supported by the source?",
                            "qualifiers": ["query"]
                        }
                    },
                    {
                        "text": {
                            "text": claim
                        }
                    }
                ]
            )

            action = response.get("action", "NONE")
            assessments = response.get("assessments", [])

            # Extract grounding score from assessments
            grounding_score = None
            for assessment in assessments:
                grounding = assessment.get("contextualGroundingPolicy", {})
                filters = grounding.get("filters", [])
                for f in filters:
                    if f.get("type") == "GROUNDING":
                        grounding_score = f.get("score", 0)
                        break

            if grounding_score is None:
                # If no grounding score in assessments, infer from action
                # GUARDRAIL_INTERVENED means the claim was NOT grounded
                grounding_score = 0.3 if action == "GUARDRAIL_INTERVENED" else 0.8

            grounded = grounding_score >= threshold

            results.append({
                "claim": claim,
                "grounding_score": round(grounding_score, 3),
                "grounded": grounded,
                "action": action,
                "verdict": "GROUNDED" if grounded else "NOT_GROUNDED",
                "method": "bedrock_contextual_grounding"
            })

        except Exception as e:
            results.append({
                "claim": claim,
                "grounding_score": 0.0,
                "grounded": False,
                "verdict": "ERROR",
                "error": str(e),
                "method": "bedrock_error"
            })

    grounded_count = sum(1 for r in results if r["grounded"])
    total = len(results)

    return {
        "summary": {
            "total_claims": total,
            "grounded": grounded_count,
            "not_grounded": total - grounded_count,
            "grounding_rate": round(grounded_count / total, 2) if total > 0 else 0,
            "threshold_used": threshold,
            "guardrail_id": guardrail_id,
            "service": "Amazon Bedrock Guardrails"
        },
        "results": results
    }


# ── MCP Server ──

TOOLS = [
    {
        "name": "bedrock_grounding_check",
        "description": (
            "Verify whether AI-generated claims are grounded in source documents "
            "using Amazon Bedrock Guardrails Contextual Grounding Check. "
            "Requires AWS credentials and a pre-configured Guardrail with grounding check enabled. "
            "Returns ML-based grounding scores (0-1) for each claim. "
            "Cost: ~$0.10 per 1,000 text units."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "claims": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of claims/assertions to verify"
                },
                "sources": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of source document texts to check claims against"
                },
                "guardrail_id": {
                    "type": "string",
                    "description": "AWS Bedrock Guardrail ID (from setup script or AWS Console)"
                },
                "guardrail_version": {
                    "type": "string",
                    "default": "DRAFT",
                    "description": "Guardrail version. Default: DRAFT"
                },
                "region": {
                    "type": "string",
                    "default": "us-east-1",
                    "description": "AWS region. Default: us-east-1"
                },
                "threshold": {
                    "type": "number",
                    "default": 0.7,
                    "description": "Minimum grounding score (0-1). Default: 0.7"
                }
            },
            "required": ["claims", "sources", "guardrail_id"]
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
            "serverInfo": {"name": "bedrock-guardrails", "version": "1.0.0"}
        }))
    elif method == "notifications/initialized":
        pass
    elif method == "tools/list":
        write_message(make_response(id, {"tools": TOOLS}))
    elif method == "tools/call":
        tool_name = params.get("name", "")
        args = params.get("arguments", {})

        if tool_name == "bedrock_grounding_check":
            try:
                result = check_grounding_bedrock(
                    claims=args["claims"],
                    sources=args["sources"],
                    guardrail_id=args["guardrail_id"],
                    guardrail_version=args.get("guardrail_version", "DRAFT"),
                    region=args.get("region", "us-east-1"),
                    threshold=args.get("threshold", 0.7)
                )
                write_message(make_response(id, {
                    "content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False, indent=2)}]
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
        result = check_grounding_bedrock(
            claims=data["claims"],
            sources=data["sources"],
            guardrail_id=data.get("guardrail_id", "981o7pz3ze8q"),
            guardrail_version=data.get("guardrail_version", "DRAFT"),
            region=data.get("region", "us-east-1"),
            threshold=data.get("threshold", 0.7),
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        asyncio.run(main())
