#!/usr/bin/env python3
"""
urlhealth MCP Server — URL Verification & Hallucination Detection
Verifies URL liveness and classifies into 4 categories:
  LIVE: HTTP 200, content accessible
  STALE: 404 but has Wayback Machine snapshot (link rot, not fabrication)
  LIKELY_HALLUCINATED: 404 and no Wayback snapshot (likely LLM-fabricated)
  UNKNOWN: Other HTTP status codes, needs manual review

Based on: arxiv 2604.03173 — Post-hoc URL verification achieves 6.4-79× URL error reduction.
"""

import asyncio
import json
import sys
import urllib.request
import urllib.error
import urllib.parse
import ssl
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


# ── URL Health Check Logic ──

def check_url_live(url: str, timeout: int = 15) -> dict:
    """Check if a URL is accessible (HTTP 200)."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) urlhealth-cli/1.0"
    }
    req = urllib.request.Request(url, headers=headers, method="GET")

    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            status = resp.getcode()
            content_type = resp.headers.get("Content-Type", "unknown")
            return {
                "reachable": True,
                "status_code": status,
                "content_type": content_type,
            }
    except urllib.error.HTTPError as e:
        return {"reachable": False, "status_code": e.code, "error": str(e.reason)}
    except urllib.error.URLError as e:
        return {"reachable": False, "status_code": 0, "error": str(e.reason)}
    except Exception as e:
        return {"reachable": False, "status_code": 0, "error": str(e)}


def check_wayback(url: str, timeout: int = 10) -> dict:
    """Check if URL has a Wayback Machine snapshot."""
    api_url = f"https://archive.org/wayback/available?url={urllib.parse.quote(url, safe='')}"
    headers = {"User-Agent": "urlhealth-cli/1.0"}
    req = urllib.request.Request(api_url, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            snapshots = data.get("archived_snapshots", {})
            closest = snapshots.get("closest", {})
            if closest.get("available"):
                return {
                    "has_snapshot": True,
                    "snapshot_url": closest.get("url", ""),
                    "snapshot_timestamp": closest.get("timestamp", ""),
                }
            return {"has_snapshot": False}
    except Exception as e:
        return {"has_snapshot": False, "error": str(e)}


def classify_url(url: str, timeout: int = 15) -> dict:
    """
    Classify a URL into one of 4 categories:
    LIVE / STALE / LIKELY_HALLUCINATED / UNKNOWN
    """
    live_result = check_url_live(url, timeout)

    if live_result["reachable"] and live_result["status_code"] == 200:
        return {
            "url": url,
            "status": "LIVE",
            "status_code": 200,
            "content_type": live_result.get("content_type", "unknown"),
            "details": "URL is accessible and returns HTTP 200",
        }

    status_code = live_result.get("status_code", 0)

    if status_code in (404, 410, 0):
        # URL not found — check Wayback Machine
        wayback = check_wayback(url)
        if wayback.get("has_snapshot"):
            return {
                "url": url,
                "status": "STALE",
                "status_code": status_code,
                "details": "URL is dead but has Wayback Machine snapshot (link rot, not fabrication)",
                "wayback_url": wayback.get("snapshot_url", ""),
                "wayback_timestamp": wayback.get("snapshot_timestamp", ""),
            }
        else:
            return {
                "url": url,
                "status": "LIKELY_HALLUCINATED",
                "status_code": status_code,
                "details": "URL is dead and has NO Wayback snapshot — likely LLM-fabricated",
            }

    # Other status codes (403, 500, 301, etc.)
    return {
        "url": url,
        "status": "UNKNOWN",
        "status_code": status_code,
        "details": f"HTTP {status_code} — may be accessible via browser. Manual review needed.",
        "error": live_result.get("error", ""),
    }


def batch_check_urls(urls: list[str], timeout: int = 15) -> dict:
    """Check multiple URLs and return summary + details."""
    results = []
    for url in urls:
        result = classify_url(url, timeout)
        results.append(result)

    # Summary
    statuses = [r["status"] for r in results]
    summary = {
        "total_urls": len(urls),
        "LIVE": statuses.count("LIVE"),
        "STALE": statuses.count("STALE"),
        "LIKELY_HALLUCINATED": statuses.count("LIKELY_HALLUCINATED"),
        "UNKNOWN": statuses.count("UNKNOWN"),
        "hallucination_rate": round(
            statuses.count("LIKELY_HALLUCINATED") / len(urls), 3
        ) if urls else 0,
    }

    return {"summary": summary, "results": results}


# ── MCP Server ──

TOOLS = [
    {
        "name": "url_health_check",
        "description": (
            "Verify URL liveness and detect likely hallucinated URLs. "
            "Classifies each URL into: LIVE (HTTP 200), STALE (dead but has Wayback Machine archive), "
            "LIKELY_HALLUCINATED (dead with no archive — likely LLM-fabricated), UNKNOWN (other status). "
            "Based on arxiv 2604.03173: post-hoc URL verification achieves 6.4-79× error reduction. "
            "Accepts a list of URLs and returns classification + summary statistics."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "urls": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of URLs to verify"
                },
                "timeout": {
                    "type": "number",
                    "default": 15,
                    "description": "HTTP request timeout in seconds. Default: 15"
                }
            },
            "required": ["urls"]
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
            "serverInfo": {"name": "urlhealth", "version": "1.0.0"}
        }))
    elif method == "notifications/initialized":
        pass
    elif method == "tools/list":
        write_message(make_response(id, {"tools": TOOLS}))
    elif method == "tools/call":
        tool_name = params.get("name", "")
        args = params.get("arguments", {})

        if tool_name == "url_health_check":
            try:
                result = batch_check_urls(
                    urls=args["urls"],
                    timeout=args.get("timeout", 15)
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
        result = batch_check_urls(
            urls=data["urls"],
            timeout=data.get("timeout", 15),
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        asyncio.run(main())
