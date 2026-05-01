"""
src/agent.py  —  Core Triage Orchestration Pipeline

Pipeline per ticket:
  1. Normalize   → clean & merge subject + issue text
  2. Risk Check  → rule-based fraud/sensitive keyword detection
  3. Company Infer → if company == None, guess from content
  4. Multi-Intent → detect if ticket has multiple distinct requests
  5. Retrieve    → TF-IDF search over support corpus
  6. Classify    → LLM classifies request_type + product_area
  7. Respond     → LLM generates grounded response or escalates
  8. Merge       → collapse multi-intent results
"""

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

# ── constants ──────────────────────────────────────────────────────────────────
MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 1500
RETRIEVAL_TOP_K = 4
MIN_CORPUS_SCORE = 0.05   # below this → escalate (no relevant docs found)

ALLOWED_STATUS       = {"replied", "escalated"}
ALLOWED_REQUEST_TYPE = {"product_issue", "feature_request", "bug", "invalid"}


class TriageAgent:
    def __init__(self, corpus: list[dict], verbose: bool = False):
        self.client       = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.retriever    = Retriever(corpus)
        self.risk_checker = RiskClassifier()
        self.verbose      = verbose

    # ──────────────────────────────────────────────────────────────────────────
    #  PUBLIC
    # ──────────────────────────────────────────────────────────────────────────
    def run(self, input_path: str, output_path: str):
        df = pd.read_csv(input_path)

        # Normalise column names (case-insensitive)
        df.columns = [c.strip().lower() for c in df.columns]
        required = {"issue"}
        if not required.issubset(set(df.columns)):
            raise ValueError(f"Input CSV must have columns: {required}. Found: {list(df.columns)}")

        # Fill optional columns
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
                time.sleep(0.3)   # respect API rate limits

        out_df = pd.DataFrame(results)
        out_df.to_csv(output_path, index=False)
        console.print(f"\n[green]Saved {len(out_df)} results → {output_path}[/green]")

        # Summary table
        self._print_summary(out_df)

    # ──────────────────────────────────────────────────────────────────────────
    #  PIPELINE STAGES
    # ──────────────────────────────────────────────────────────────────────────
    def _process_row(self, idx: int, row: pd.Series) -> dict:
        issue   = str(row.get("issue",   "") or "").strip()
        subject = str(row.get("subject", "") or "").strip()
        company = str(row.get("company", "") or "").strip()

        # 1. Normalize
        full_text = self._normalize(issue, subject)

        # 2. Risk check (rule-based — fast, no LLM call)
        risk = self.risk_checker.check(full_text)

        # 3. Infer company if None / empty
        if not company or company.lower() in ("none", "nan", ""):
            company = self._infer_company(full_text)

        # 4. Multi-intent detection
        intents = self._detect_intents(full_text)

        # 5. Retrieve corpus docs (per inferred company + combined text)
        docs = self.retriever.search(
            query=full_text,
            company_filter=company if company.lower() not in ("none", "") else None,
            top_k=RETRIEVAL_TOP_K,
        )
        corpus_found = bool(docs) and docs[0]["score"] >= MIN_CORPUS_SCORE

        # 6 + 7.  Classify & Respond via LLM
        result = self._llm_triage(
            full_text=full_text,
            company=company,
            intents=intents,
            docs=docs,
            corpus_found=corpus_found,
            risk=risk,
        )

        # 8. Validate output fields
        result = self._validate(result)
        result["_row_idx"] = idx
        return result

    # ── normalize ──────────────────────────────────────────────────────────────
    @staticmethod
    def _normalize(issue: str, subject: str) -> str:
        parts = []
        if subject and subject.lower() not in ("nan", "none", "n/a", ""):
            parts.append(f"Subject: {subject}")
        if issue:
            parts.append(issue)
        text = " | ".join(parts)
        # Strip control chars
        text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", " ", text)
        return text.strip()

    # ── infer company ──────────────────────────────────────────────────────────
    def _infer_company(self, text: str) -> str:
        text_lower = text.lower()
        scores = {"HackerRank": 0, "Claude": 0, "Visa": 0}

        # HackerRank signals
        for kw in ("hackerrank", "test", "coding challenge", "assessment", "hiring",
                   "interview", "proctoring", "plagiarism", "leaderboard", "rank"):
            if kw in text_lower:
                scores["HackerRank"] += 1

        # Claude signals
        for kw in ("claude", "anthropic", "ai response", "ai model", "hallucination",
                   "conversation", "prompt", "context window", "claude.ai"):
            if kw in text_lower:
                scores["Claude"] += 1

        # Visa signals
        for kw in ("visa", "card", "payment", "transaction", "merchant", "bank",
                   "charge", "refund", "fraud", "chargeback", "cvv", "pin", "atm"):
            if kw in text_lower:
                scores["Visa"] += 1

        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else "None"

    # ── multi-intent detection ─────────────────────────────────────────────────
    def _detect_intents(self, text: str) -> list[str]:
        """
        Use LLM to split compound tickets into individual intents.
        Falls back to [text] on any error.
        """
        if len(text) < 50:   # too short to be multi-intent
            return [text]

        try:
            resp = self.client.messages.create(
                model=MODEL,
                max_tokens=400,
                system=MULTI_INTENT_PROMPT,
                messages=[{"role": "user", "content": text}],
            )
            raw = resp.content[0].text.strip()
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
        intents: list[str],
        docs: list[dict],
        corpus_found: bool,
        risk: dict,
    ) -> dict:
        """
        Single Claude call that classifies + generates a grounded response.
        """
        corpus_context = self._format_corpus(docs) if corpus_found else "No relevant documentation found."

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

            # Strip markdown fences if present
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

            result = json.loads(raw)
            return result

        except json.JSONDecodeError as e:
            console.print(f"[yellow]JSON parse error row — using fallback escalation[/yellow]")
            return self._fallback_escalation(full_text, reason=f"LLM output parse error: {e}")
        except Exception as e:
            console.print(f"[red]API error: {e}[/red]")
            return self._fallback_escalation(full_text, reason=str(e))

    # ── helpers ────────────────────────────────────────────────────────────────
    @staticmethod
    def _format_corpus(docs: list[dict]) -> str:
        if not docs:
            return "No relevant documentation found."
        sections = []
        for i, d in enumerate(docs, 1):
            sections.append(
                f"[Doc {i} | Source: {d['source']} | Score: {d['score']:.2f}]\n{d['text']}"
            )
        return "\n\n---\n\n".join(sections)

    @staticmethod
    def _fallback_escalation(text: str, reason: str) -> dict:
        return {
            "status":       "escalated",
            "product_area": "unknown",
            "response":     "We were unable to process your request automatically. A support agent will reach out shortly.",
            "justification": f"Fallback escalation due to: {reason}",
            "request_type": "product_issue",
        }

    @staticmethod
    def _validate(result: dict) -> dict:
        """Clamp output fields to allowed values."""
        if result.get("status") not in ALLOWED_STATUS:
            result["status"] = "escalated"
        if result.get("request_type") not in ALLOWED_REQUEST_TYPE:
            result["request_type"] = "product_issue"
        for key in ("response", "justification", "product_area"):
            if not result.get(key):
                result[key] = "N/A"
        return result

    # ── display ────────────────────────────────────────────────────────────────
    def _print_result(self, idx: int, result: dict):
        color = "red" if result["status"] == "escalated" else "green"
        console.print(
            f"\n[bold]Ticket {idx + 1}[/bold] | "
            f"[{color}]{result['status'].upper()}[/{color}] | "
            f"{result['request_type']} | "
            f"{result['product_area']}"
        )
        console.print(f"  Response: {result['response'][:120]}...")
        console.print(f"  Reason:   {result['justification'][:100]}...")

    @staticmethod
    def _print_summary(df: pd.DataFrame):
        table = Table(title="Triage Summary", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", justify="right")

        total      = len(df)
        replied    = (df["status"] == "replied").sum()
        escalated  = (df["status"] == "escalated").sum()

        table.add_row("Total Tickets",   str(total))
        table.add_row("Replied",         f"[green]{replied}[/green]")
        table.add_row("Escalated",       f"[red]{escalated}[/red]")

        if "request_type" in df.columns:
            for rt in ["bug", "product_issue", "feature_request", "invalid"]:
                count = (df["request_type"] == rt).sum()
                if count:
                    table.add_row(f"  → {rt}", str(count))

        console.print("\n")
        console.print(table)