#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capture vLLM Rust parser output for the Unified conformance tab.

vLLM's native Rust `UnifiedParser` (crate `vllm-parser`, module `unified`) is NOT
exposed to Python (the PyO3 bindings only bind tool parsers), so — like
capture_vllm_rust.py — this builds a small temporary Rust binary that depends on the
`vllm-parser` + `vllm-tokenizer` crates from a checked-out vLLM source tree and feeds
the cases through the right unified parser per family:

  * gemma4/kimi_k3 -> native UnifiedParser
  * qwen3/kimi_k2/deepseek_v4/deepseek_v41/glm47 -> CombinedParser(reasoning, tool)

Both emit ordered `UnifiedParserEvent { Text | Reasoning | ToolCall }`, which is exactly
the golden event schema. Output JSON: {"vllm_rust_version", "results": {id: {assembled,
chunks, parser}}}. Runs on the HOST (needs cargo + the vLLM rust source).

Usage:
  python3 capture_vllm_rust_unified.py --vllm-rust-source /path/to/vllm-0.30.0/rust \
      --job job.json --out conformance/unified/vllm_rust_capture.yaml
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import tomllib
import yaml
import sys
import tempfile
from pathlib import Path

from capture_stimulus import capture_peer_results
from unified_tools import SCHEMA_PATH

# Family parser wiring for the released 0.30.0 capture.
FAMILY_PARSERS = {
    "gemma4": ("unified", None, None),
    "kimi_k3": ("unified", None, None),
    "deepseek_v4": ("combined", "DeepSeekV4ReasoningParser", "DeepSeekV4ToolParser"),
    "deepseek_v41": ("combined", "DeepSeekV41ReasoningParser", "DeepSeekV41ToolParser"),
    "glm47": ("combined", "Glm47ReasoningParser", "Glm47MoeToolParser"),
    "qwen3": ("combined", "Qwen3ReasoningParser", "Qwen3CoderToolParser"),
    "kimi_k2": ("combined", "KimiK2ReasoningParser", "KimiK2ToolParser"),
}

RUST_MAIN = r'''
use std::collections::BTreeMap;
use std::io::Read;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use vllm_parser::reasoning::{
    DeepSeekV4ReasoningParser, DeepSeekV41ReasoningParser, Glm47ReasoningParser,
    KimiK2ReasoningParser, Qwen3ReasoningParser, ReasoningParser,
};
use vllm_parser::tool::{
    DeepSeekV4ToolParser, DeepSeekV41ToolParser, Glm47MoeToolParser,
    KimiK2ToolParser, Qwen3CoderToolParser, Tool, ToolParser,
};
use vllm_parser::unified::{
    CombinedParser, Gemma4UnifiedParser, KimiK3UnifiedParser, UnifiedParser, UnifiedParserEvent, UnifiedParserOutput,
};
use vllm_tokenizer::test_utils::TestTokenizer;
use vllm_tokenizer::{DecodedText, DynTokenizer};

#[derive(Deserialize)]
struct Job {
    cases: Vec<Case>,
}
#[derive(Deserialize)]
struct Case {
    id: String,
    family: String,
    input: String,
    #[serde(default)]
    chunks: Vec<String>,
    #[serde(default)]
    terminal_step: bool,
    #[serde(default = "tools")]
    tools: Vec<Tool>,
}

#[derive(Serialize)]
struct CaseOut {
    assembled: Vec<Value>,
    chunks: Vec<Vec<Value>>,
    parser: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

fn tools() -> Vec<Tool> {
    serde_json::from_str(include_str!("unified_tools.json"))
        .expect("Unified corpus tool schemas")
}

fn make_parser(family: &str, tools: &[Tool]) -> (Box<dyn UnifiedParser>, String) {
    match family {
        "gemma4" => {
            let tok: DynTokenizer = Arc::new(
                TestTokenizer::new()
                    .with_special_token("<|channel>", 256)
                    .with_special_token("<channel|>", 257),
            );
            (
                Gemma4UnifiedParser::create(tools, tok).expect("gemma4 unified create"),
                "vLLM Rust (UnifiedParser)".to_string(),
            )
        }
        "qwen3" => {
            let tok: DynTokenizer = Arc::new(
                TestTokenizer::new()
                    .with_regular_token("<think>", 256)
                    .with_regular_token("</think>", 257),
            );
            let reasoning = Qwen3ReasoningParser::create(tok).expect("qwen3 reasoning");
            let tool = Qwen3CoderToolParser::create(tools).expect("qwen3 tool");
            (
                Box::new(CombinedParser::new(Some(reasoning), Some(tool))),
                "vLLM Rust (CombinedParser)".to_string(),
            )
        }
        "kimi_k2" => {
            let tok: DynTokenizer = Arc::new(
                TestTokenizer::new()
                    .with_special_token("<think>", 256)
                    .with_special_token("</think>", 257),
            );
            let reasoning = KimiK2ReasoningParser::create(tok).expect("kimi reasoning");
            let tool = KimiK2ToolParser::create(tools).expect("kimi tool");
            (
                Box::new(CombinedParser::new(Some(reasoning), Some(tool))),
                "vLLM Rust (CombinedParser)".to_string(),
            )
        }
        "kimi_k3" => {
            let tok: DynTokenizer = Arc::new(
                TestTokenizer::new()
                    .with_special_token("<|open|>", 256)
                    .with_special_token("<|sep|>", 257),
            );
            (
                KimiK3UnifiedParser::create(tools, tok).expect("kimi k3 unified create"),
                "vLLM Rust (UnifiedParser)".to_string(),
            )
        }
        "deepseek_v4" | "deepseek_v41" | "glm47" => {
            let tok: DynTokenizer = Arc::new(
                TestTokenizer::new()
                    .with_special_token("<think>", 256)
                    .with_special_token("</think>", 257),
            );
            let (reasoning, tool) = match family {
                "deepseek_v4" => (
                    DeepSeekV4ReasoningParser::create(tok).expect("deepseek v4 reasoning"),
                    DeepSeekV4ToolParser::create(tools).expect("deepseek v4 tool"),
                ),
                "deepseek_v41" => (
                    DeepSeekV41ReasoningParser::create(tok).expect("deepseek v41 reasoning"),
                    DeepSeekV41ToolParser::create(tools).expect("deepseek v41 tool"),
                ),
                _ => (
                    Glm47ReasoningParser::create(tok).expect("glm47 reasoning"),
                    Glm47MoeToolParser::create(tools).expect("glm47 tool"),
                ),
            };
            (
                Box::new(CombinedParser::new(Some(reasoning), Some(tool))),
                "vLLM Rust (CombinedParser)".to_string(),
            )
        }
        other => panic!("no vLLM Rust unified mapping for family `{other}`"),
    }
}

/// Coalesce an ordered event list into JSON [reasoning|text|tool_call], merging
/// per-`tool_index` ToolCall deltas into one call (mirrors the Dynamo feed()).
fn events_to_json(events: &[UnifiedParserEvent]) -> Vec<Value> {
    let mut out: Vec<Value> = Vec::new();
    let mut slots: BTreeMap<usize, usize> = BTreeMap::new();
    let mut names: BTreeMap<usize, String> = BTreeMap::new();
    let mut raw_args: BTreeMap<usize, String> = BTreeMap::new();
    for ev in events {
        match ev {
            UnifiedParserEvent::Reasoning(t) => out.push(json!({"kind":"reasoning","text":t.text})),
            UnifiedParserEvent::Text(t) => out.push(json!({"kind":"text","text":t})),
            UnifiedParserEvent::ToolCall(d) => {
                slots.entry(d.tool_index).or_insert_with(|| {
                    out.push(json!({"kind":"tool_call","name":"","arguments":{}}));
                    out.len() - 1
                });
                if let Some(n) = &d.name {
                    names.entry(d.tool_index).or_default().push_str(n);
                }
                raw_args.entry(d.tool_index).or_default().push_str(&d.arguments);
            }
        }
    }
    for (ti, pos) in &slots {
        let name = names.get(ti).cloned().unwrap_or_default();
        let raw = raw_args.get(ti).cloned().unwrap_or_default();
        let args: Value = if raw.trim().is_empty() {
            json!({})
        } else {
            serde_json::from_str(&raw).unwrap_or(Value::String(raw))
        };
        out[*pos] = json!({"kind":"tool_call","name":name,"arguments":args});
    }
    out
}

/// Raw per-chunk deltas (not coalesced), matching the Dynamo chunk feed shape.
fn deltas_to_json(events: &[UnifiedParserEvent]) -> Vec<Value> {
    events
        .iter()
        .map(|ev| match ev {
            UnifiedParserEvent::Reasoning(t) => json!({"kind":"reasoning","text":t.text}),
            UnifiedParserEvent::Text(t) => json!({"kind":"text","text":t}),
            UnifiedParserEvent::ToolCall(d) => {
                json!({"kind":"tool_call","name":d.name,"arguments":d.arguments})
            }
        })
        .collect()
}

fn main() {
    let mut buf = String::new();
    std::io::stdin().read_to_string(&mut buf).unwrap();
    let job: Job = serde_json::from_str(&buf).unwrap();
    let mut results: BTreeMap<String, CaseOut> = BTreeMap::new();

    for case in &job.cases {
        let (mut p, parser) = make_parser(&case.family, &case.tools);
        let mut error: Option<String> = None;

        // Batch: whole input -> assembled events.
        let mut out = UnifiedParserOutput::default();
        if let Err(e) = p.parse_into(DecodedText::unattributed(case.input.clone()), &mut out) {
            error = Some(format!("UnifiedParserError::{e:?}"));
        }
        match p.finish() {
            Ok(fin) => out.events.extend(fin.events),
            Err(e) => {
                error.get_or_insert_with(|| format!("UnifiedParserError::{e:?}"));
            }
        };
        let assembled = events_to_json(&out.events);

        // Streaming: fresh parser, per-chunk deltas.
        let (mut ps, _) = make_parser(&case.family, &case.tools);
        let mut chunk_rows: Vec<Vec<Value>> = Vec::new();
        for (i, ch) in case.chunks.iter().enumerate() {
            let mut co = UnifiedParserOutput::default();
            if let Err(e) = ps.parse_into(DecodedText::unattributed(ch.clone()), &mut co) {
                error.get_or_insert_with(|| format!("UnifiedParserError::{e:?}"));
            }
            if !case.terminal_step && i == case.chunks.len() - 1 {
                match ps.finish() {
                    Ok(fin) => co.events.extend(fin.events),
                    Err(e) => {
                        error.get_or_insert_with(|| format!("UnifiedParserError::{e:?}"));
                    }
                }
            }
            chunk_rows.push(deltas_to_json(&co.events));
        }
        if case.terminal_step {
            match ps.finish() {
                Ok(fin) => chunk_rows.push(deltas_to_json(&fin.events)),
                Err(e) => {
                    error.get_or_insert_with(|| format!("UnifiedParserError::{e:?}"));
                    chunk_rows.push(Vec::new());
                }
            }
        }

        results.insert(
            case.id.clone(),
            CaseOut { assembled, chunks: chunk_rows, parser, error },
        );
    }

    let feed = json!({"vllm_rust_version": "__VLLM_VERSION__", "results": results});
    println!("{}", serde_json::to_string(&feed).unwrap());
}
'''


def _vllm_rust_version(vllm_rust_source: Path, parser_crate: Path) -> str:
    """Read the version from the checked-out vLLM Rust workspace, never a literal."""
    tag = subprocess.run(
        ["git", "describe", "--tags", "--exact-match", "HEAD"],
        cwd=vllm_rust_source.parent, capture_output=True, text=True,
    )
    if tag.returncode == 0 and re.fullmatch(r"v?\d+\.\d+\.\d+", tag.stdout.strip()):
        return tag.stdout.strip().removeprefix("v")
    parser_manifest = tomllib.loads((parser_crate / "Cargo.toml").read_text())
    version = (parser_manifest.get("package") or {}).get("version")
    if isinstance(version, str):
        return version
    for parent in (parser_crate, *parser_crate.parents):
        if parent == vllm_rust_source.parent:
            break
        manifest = parent / "Cargo.toml"
        if not manifest.exists():
            continue
        workspace_version = (tomllib.loads(manifest.read_text()).get("workspace", {})
                             .get("package", {}).get("version"))
        if isinstance(workspace_version, str):
            return workspace_version
    raise ValueError(f"could not determine vLLM Rust version below {vllm_rust_source}")


def build_and_run(vllm_rust_source: Path, job_json: str) -> str:
    """Create a temp crate depending on the vLLM rust parser/tokenizer crates, build,
    and run it with the job on stdin. Returns the captured stdout JSON."""
    parser_crate = vllm_rust_source / "src/parser"
    tok_crate = vllm_rust_source / "src/tokenizer"
    for p in (parser_crate / "Cargo.toml", tok_crate / "Cargo.toml"):
        if not p.exists():
            sys.exit(f"vLLM rust crate not found: {p}")
    version = _vllm_rust_version(vllm_rust_source, parser_crate)

    with tempfile.TemporaryDirectory(prefix="vllm-rust-uni-") as td:
        crate = Path(td)
        (crate / "src").mkdir()
        (crate / "src/unified_tools.json").write_bytes(SCHEMA_PATH.read_bytes())
        (crate / "Cargo.toml").write_text(f'''[package]
name = "vllm-rust-unified-capture"
version = "0.0.0"
edition = "2024"

[dependencies]
vllm-parser = {{ path = "{parser_crate}" }}
vllm-tokenizer = {{ path = "{tok_crate}", features = ["test-utils"] }}
serde = {{ version = "1", features = ["derive"] }}
serde_json = "1"

[workspace]
''')
        (crate / "src/main.rs").write_text(
            RUST_MAIN.replace('"vllm_rust_version": "__VLLM_VERSION__"',
                              f'"vllm_rust_version": {json.dumps(version)}')
        )
        build = subprocess.run(
            ["cargo", "build", "--release", "--quiet"],
            cwd=crate, capture_output=True, text=True)
        if build.returncode != 0:
            sys.exit(f"cargo build failed:\n{build.stderr}")
        run = subprocess.run(
            [str(crate / "target/release/vllm-rust-unified-capture")],
            input=job_json, capture_output=True, text=True)
        if run.returncode != 0:
            sys.exit(f"capture run failed:\n{run.stderr}")
        return run.stdout


def capture_job(vllm_rust_source, job):
    feed = {}
    schema_bytes = SCHEMA_PATH.read_bytes()

    def capture(cases):
        feed.update(json.loads(build_and_run(vllm_rust_source, json.dumps({"cases": cases}))))
        return feed["results"]

    results = capture_peer_results(job.get("cases", []), FAMILY_PARSERS, capture,
                                   tools=json.loads(schema_bytes), supports_finish=True, supports_tools=True)
    if SCHEMA_PATH.read_bytes() != schema_bytes:
        raise ValueError("tool schema changed during peer capture")
    if not feed:
        feed["vllm_rust_version"] = _vllm_rust_version(vllm_rust_source, vllm_rust_source / "src/parser")
    return {**feed, "results": results}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm-rust-source", required=True, type=Path)
    ap.add_argument("--job", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    data = capture_job(args.vllm_rust_source, json.loads(args.job.read_text()))
    args.out.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False,
                                  allow_unicode=True, width=4096))
    print(f"wrote {args.out} "
          f"(vllm_rust_version={data.get('vllm_rust_version')}, "
          f"results={len(data['results'])})")


if __name__ == "__main__":
    main()
