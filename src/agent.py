"""
src/agent.py  —  Core Triage Orchestration Pipeline

Pipeline per ticket:
  1. Normalize    → merge subject + issue, strip noise
  2. Risk Check   → rule-based pre-filter (ZERO LLM cost)
  3. Company Infer → guess domain if company == None
  4. Multi-Intent → split compound tickets
  5. Retrieve     → TF-IDF corpus search
  6+7. LLM Triage → classify + generate grounded response
  8. Validate     → clamp fields to allowed values
"""

from __future__ import annotations

import os
import json
import re
import time
import pandas as pd
import anthropic
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from src.retriever import Retriever
from src.classifier import RiskClassifier
from src.prompts import (
    TRIAGE_SYSTEM_PROMPT,
    build_triage_user_prompt,
    MULTI_INTENT_PROMPT,
)

console = Console()

MODEL            = "claude-sonnet-4-20250514"
MAX_TOKENS       = 1500
RETRIEVAL_TOP_K  = 5

# FIXED: lowered from 0.05 → 0.01
# The built-in corpus is small, so TF-IDF scores are naturally low (0.01–0.04).
# 0.05 was causing every ticket to fail the threshold and auto-escalate.
MIN_CORPUS_SCORE = 0.01

ALLOWED_STATUS       = {"replied", "escalated"}
ALLOWED_REQUEST_TYPE = {"product_issue", "feature_request", "bug", "invalid"}


class TriageAgent:
    def __init__(self, corpus: list, verbose: bool = False):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY not set.\n"
                "Run:  export ANTHROPIC_API_KEY='sk-ant-...'"
            )
        self.client       = anthropic.Anthropic(api_key=api_key)
        self.retriever    = Retriever(corpus)
        self.risk_checker = RiskClassifier()
        self.verbose      = verbose

    # ─────────────────────────────────────────────────────────────────────────
    def run(self, input_path: str, output_path: str):
        df = pd.read_csv(input_path)
        df.columns = [c.strip().lower() for c in df.columns]

        if df.empty:
            console.print(
                "[red]ERROR:[/red] The CSV is empty!\n"
                "Add rows with columns:  issue, subject, company\n"
                "Use data/sample_support_issues.csv as a reference."
            )
            return

        if "issue" not in df.columns:
            raise ValueError(
                f"Input CSV must have an 'issue' column. Found: {list(df.columns)}\n"
                "Expected header:  issue,subject,company"
            )

        for col in ("subject", "company"):
            if col not in df.columns:
                df[col] = ""

        results = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Triaging tickets...", total=len(df))
            for idx, row in df.iterrows():
                progress.update(task, description=f"[cyan]Ticket {idx + 1}/{len(df)}")
                result = self._process_row(idx, row)
                results.append(result)
                if self.verbose:
                    self._print_result(idx, result)
                progress.advance(task)
                time.sleep(0.3)

        out_df = pd.DataFrame(results)
        if "_row_idx" in out_df.columns:
            out_df = out_df.drop(columns=["_row_idx"])

        out_df.to_csv(output_path, index=False)
        console.print(f"\n[green]Saved {len(out_df)} results → {output_path}[/green]")
        self._print_summary(out_df)

    # ─────────────────────────────────────────────────────────────────────────
    def _process_row(self, idx: int, row: pd.Series) -> dict:
        issue   = str(row.get("issue",   "") or "").strip()
        subject = str(row.get("subject", "") or "").strip()
        company = str(row.get("company", "") or "").strip()

        full_text = self._normalize(issue, subject)

        if not full_text:
            return {**self._fallback_escalation("", "Empty ticket body"), "_row_idx": idx}

        # Step 2: Risk check
        risk = self.risk_checker.check(full_text)

        # Step 3: Infer company
        if not company or company.lower() in ("none", "nan", ""):
            company = self._infer_company(full_text)

        # Step 4: Multi-intent split
        intents = self._detect_intents(full_text)

        # Step 5: Corpus retrieval
        docs = self.retriever.search(
            query=full_text,
            company_filter=company if company.lower() not in ("none", "") else None,
            top_k=RETRIEVAL_TOP_K,
        )
        corpus_found = bool(docs) and docs[0]["score"] >= MIN_CORPUS_SCORE

        # If prompt injection detected → skip LLM, hard escalate immediately
        if risk["is_injected"]:
            result = {
                "status":        "escalated",
                "product_area":  "security",
                "response":      "This request could not be processed. Please contact support directly.",
                "justification": "Prompt injection attempt detected by pre-classifier. Immediate escalation.",
                "request_type":  "invalid",
            }
        else:
            # Steps 6+7: LLM triage
            result = self._llm_triage(
                full_text=full_text,
                company=company,
                intents=intents,
                docs=docs,
                corpus_found=corpus_found,
                risk=risk,
            )

        result = self._validate(result)
        result["_row_idx"] = idx
        return result

    # ── normalize ─────────────────────────────────────────────────────────────
    @staticmethod
    def _normalize(issue: str, subject: str) -> str:
        parts = []
        if subject and subject.lower() not in ("nan", "none", "n/a", ""):
            parts.append(f"Subject: {subject}")
        if issue:
            parts.append(issue)
        text = " | ".join(parts)
        text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", " ", text)
        return text.strip()

    # ── infer company ─────────────────────────────────────────────────────────
    def _infer_company(self, text: str) -> str:
        t = text.lower()
        scores = {"HackerRank": 0, "Claude": 0, "Visa": 0}
        for kw in ("hackerrank", "coding challenge", "assessment", "interview",
                   "proctoring", "plagiarism", "leaderboard", "test case",
                   "submission", "hire", "recruiter"):
            if kw in t:
                scores["HackerRank"] += 2
        for kw in ("claude", "anthropic", "ai response", "hallucination",
                   "context window", "claude.ai", "ai model", "llm"):
            if kw in t:
                scores["Claude"] += 2
        for kw in ("visa", "card", "payment", "transaction",
                   "charge", "refund", "fraud", "chargeback", "cvv", "atm",
                   "bank", "merchant", "declined", "pin"):
            if kw in t:
                scores["Visa"] += 2
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else "None"

    # ── multi-intent detection ────────────────────────────────────────────────
    def _detect_intents(self, text: str) -> list:
        if len(text) < 60:
            return [text]
        try:
            resp = self.client.messages.create(
                model=MODEL,
                max_tokens=400,
                system=MULTI_INTENT_PROMPT,
                messages=[{"role": "user", "content": text}],
            )
            raw    = resp.content[0].text.strip()
            parsed = json.loads(raw)
            intents = parsed.get("intents", [text])
            return intents if isinstance(intents, list) and intents else [text]
        except Exception:
            return [text]

    # ── LLM triage ────────────────────────────────────────────────────────────
    def _llm_triage(
        self,
        full_text: str,
        company: str,
        intents: list,
        docs: list,
        corpus_found: bool,
        risk: dict,
    ) -> dict:
        corpus_context = (
            self._format_corpus(docs) if corpus_found
            else "No highly relevant documents found in corpus. Use general domain knowledge carefully or escalate."
        )
        user_prompt = build_triage_user_prompt(
            issue_text=full_text,
            company=company,
            intents=intents,
            corpus_context=corpus_context,
            corpus_found=corpus_found,
            risk=risk,
        )
        try:
            resp = self.client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=TRIAGE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            raw = resp.content[0].text.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            return json.loads(raw)
        except json.JSONDecodeError as e:
            console.print("[yellow]  ⚠ JSON parse error — escalating[/yellow]")
            return self._fallback_escalation(full_text, f"LLM parse error: {e}")
        except Exception as e:
            console.print(f"[red]  ✗ API error: {e}[/red]")
            return self._fallback_escalation(full_text, str(e))

    # ── helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _format_corpus(docs: list) -> str:
        if not docs:
            return "No relevant documentation found."
        sections = []
        for i, d in enumerate(docs, 1):
            sections.append(
                f"[Doc {i} | Source: {d['source']} | Score: {d['score']:.3f}]\n{d['text']}"
            )
        return "\n\n---\n\n".join(sections)

    @staticmethod
    def _fallback_escalation(text: str, reason: str) -> dict:
        return {
            "status":        "escalated",
            "product_area":  "unknown",
            "response":      "We were unable to process your request. A support agent will reach out shortly.",
            "justification": f"Fallback escalation: {reason}",
            "request_type":  "product_issue",
        }

    @staticmethod
    def _validate(result: dict) -> dict:
        if result.get("status") not in ALLOWED_STATUS:
            result["status"] = "escalated"
        if result.get("request_type") not in ALLOWED_REQUEST_TYPE:
            result["request_type"] = "product_issue"
        for key in ("response", "justification", "product_area"):
            if not result.get(key):
                result[key] = "N/A"
        return result

    # ── display ───────────────────────────────────────────────────────────────
    def _print_result(self, idx: int, result: dict):
        color = "red" if result["status"] == "escalated" else "green"
        console.print(
            f"\n[bold]#{idx + 1}[/bold] [{color}]{result['status'].upper()}[/{color}]"
            f" | {result['request_type']} | {result['product_area']}"
        )
        console.print(f"  Response:      {result['response'][:120]}")
        console.print(f"  Justification: {result['justification'][:100]}")

    @staticmethod
    def _print_summary(df: pd.DataFrame):
        table = Table(title="Triage Summary", header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", justify="right")
        total     = len(df)
        replied   = (df["status"] == "replied").sum()
        escalated = (df["status"] == "escalated").sum()
        table.add_row("Total Tickets", str(total))
        table.add_row("Replied",       f"[green]{replied}[/green]")
        table.add_row("Escalated",     f"[red]{escalated}[/red]")
        if "request_type" in df.columns:
            for rt in ["bug", "product_issue", "feature_request", "invalid"]:
                count = (df["request_type"] == rt).sum()
                if count:
                    table.add_row(f"  → {rt}", str(count))
        console.print("\n")
        console.print(table)