# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Live-capture vLLM 0.30.0 parser output for the Unified conformance tab.

Runs with an installed vLLM package or a release source checkout (`import vllm`). Reads a JSON job on stdin:

    {"cases": [{"id": "...", "family": "gemma4", "input": "...",
                "chunks": ["<chunk1>", "<chunk2>", ...]}]}

and writes on stdout, per case:

    {"results": {"<id>": {
        "assembled": [ {kind: reasoning|text|tool_call, ...} ],   # batch parse()
        "chunks":    [ [ {kind,...}, ... ], ... ]                  # per-chunk parse_delta
    }}}

Batch `parse()` gives the real FINAL-MESSAGE fields (reasoning, content, tools)
projected to an ordered event list; streaming `parse_delta` gives the real
per-chunk deltas. No GPU / model needed — the parser only lexes text, so a stub
tokenizer (empty vocab -> markers matched as text) is enough.
"""
import json
import sys
import subprocess
from pathlib import Path
import yaml

from capture_stimulus import capture_peer_results
from unified_tools import unified_tools

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser.parser_manager import ParserManager

# family -> (reasoning_parser_name, tool_parser_name) in vLLM 0.30.0.
FAMILY_PARSERS = {
    "gemma4": ("gemma4", "gemma4"),
    "deepseek_v4": ("deepseek_v4", "deepseek_v4"),
    "deepseek_v41": ("deepseek_v41", "deepseek_v41"),
    "kimi_k3": ("kimi_k3", "kimi_k3"),
    "muse_glimmer": ("muse_glimmer", "muse_glimmer"),
    "qwen3": ("qwen3", "qwen3_coder"),
    "glm47": ("glm47", "glm47"),
    "kimi_k2": ("kimi_k2", "kimi_k2"),
}

TOOLS = [
    {"type": "function", "function": tool} for tool in unified_tools()
]


class StubTokenizer:
    """Text-only tokenizer: empty vocab so every marker is matched as text."""
    all_special_tokens = []
    is_fast = True

    def get_vocab(self):
        return {}

    def convert_tokens_to_ids(self, t):
        return None

    def get_added_vocab(self):
        return {}

    def encode(self, t, add_special_tokens=False):
        return []

    def decode(self, ids, **k):
        return ""

    @property
    def vocab_size(self):
        return 0


def _fn(tc):
    return getattr(tc, "function", tc)


def _tool_args_json(args):
    if not args:
        return {}
    try:
        return json.loads(args)
    except (ValueError, TypeError):
        return args


def _assembled_events(reasoning, content, tool_calls):
    """Project vLLM's (reasoning, content, tool_calls) final message to events."""
    events = []
    if reasoning:
        events.append({"kind": "reasoning", "text": reasoning})
    if content:
        events.append({"kind": "text", "text": content})
    for tc in tool_calls or []:
        fn = _fn(tc)
        events.append({"kind": "tool_call",
                       "name": getattr(fn, "name", None) or "",
                       "arguments": _tool_args_json(getattr(fn, "arguments", None))})
    return events


def _delta_events(dm):
    """One parse_delta DeltaMessage -> raw per-chunk unified deltas."""
    out = []
    if dm is None:
        return out
    if getattr(dm, "reasoning", None):
        out.append({"kind": "reasoning", "text": dm.reasoning})
    if getattr(dm, "content", None):
        out.append({"kind": "text", "text": dm.content})
    for tc in getattr(dm, "tool_calls", None) or []:
        fn = _fn(tc)
        out.append({"kind": "tool_call",
                    "name": getattr(fn, "name", None),
                    "arguments": getattr(fn, "arguments", None)})
    return out


def _capture_cases(cases):
    mgr = ParserManager()
    results = {}
    for case in cases:
        fam = case["family"]
        if fam not in FAMILY_PARSERS:
            continue
        req = ChatCompletionRequest(
            messages=[{"role": "user", "content": "x"}],
            tools=[{"type": "function", "function": tool}
                   for tool in case.get("tools", unified_tools())],
            tool_choice="auto",
        )
        rn, tn = FAMILY_PARSERS[fam]
        cls = mgr.get_parser(tool_parser_name=tn, reasoning_parser_name=rn,
                             enable_auto_tools=True, model_name=fam)
        if cls is None:
            results[case["id"]] = {"unavailable": f"vLLM parser manager has no parser for {fam}"}
            continue

        try:
            # Batch: the real final-message fields, projected to an ordered list.
            reasoning, content, tool_calls = cls(StubTokenizer(), tools=req.tools).parse(
                case["input"], req, True, [])
            assembled = _assembled_events(reasoning, content, tool_calls)

            # Streaming: real per-chunk deltas.
            p = cls(StubTokenizer(), tools=req.tools)
            if hasattr(p, "initialize_streaming"):
                p.initialize_streaming()
            chunks = case.get("chunks", [])
            per_chunk = []
            for i, ch in enumerate(chunks):
                dm = p.parse_delta(ch, [], req, [], finished=(not case["terminal_step"] and i == len(chunks) - 1))
                per_chunk.append(_delta_events(dm))
            if case["terminal_step"]:
                per_chunk.append(_delta_events(p.parse_delta("", [], req, [], finished=True)))

            results[case["id"]] = {"assembled": assembled, "chunks": per_chunk}
        except Exception as exc:
            results[case["id"]] = {"error": f"{type(exc).__name__}: {exc}"}

    return results


def main():
    job = json.load(sys.stdin)
    results = capture_peer_results(job.get("cases", []), FAMILY_PARSERS, _capture_cases,
                                   tools=[tool["function"] for tool in TOOLS], supports_finish=True, supports_tools=True)

    # YAML to match the conformance fixture corpus. Container stdout is log-polluted,
    # so a recapture writes this to a file (or strips lines before the first top-level
    # key) rather than grepping a single JSON line.
    yaml.dump({"results": results, "vllm_version": _vllm_version()}, sys.stdout,
              default_flow_style=False, sort_keys=False, allow_unicode=True, width=4096)


def _vllm_version():
    import vllm
    if vllm.__version__ != "dev":
        return vllm.__version__
    # Source checkouts do not have the build-generated vllm/_version.py.
    return subprocess.check_output(
        ["git", "describe", "--tags", "--exact-match", "HEAD"],
        cwd=Path(vllm.__file__).resolve().parent, text=True,
    ).strip().removeprefix("v")


if __name__ == "__main__":
    main()
