from __future__ import annotations

import re
import time
import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from src.retriever import Retriever
from src.classifier import RiskClassifier

console = Console()

RETRIEVAL_TOP_K = 5
MIN_CORPUS_SCORE = 0.01

ALLOWED_STATUS = {"replied", "escalated"}
ALLOWED_REQUEST_TYPE = {"product_issue", "feature_request", "bug", "invalid"}


class TriageAgent:
    def __init__(self, corpus: list, verbose: bool = False):
        self.retriever = Retriever(corpus)
        self.risk_checker = RiskClassifier()
        self.verbose = verbose

    # ─────────────────────────────────────────────────────────
    def run(self, input_path: str, output_path: str):
        df = pd.read_csv(input_path)
        df.columns = [c.strip().lower() for c in df.columns]

        if "issue" not in df.columns:
            raise ValueError("CSV must contain 'issue' column")

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
            task = progress.add_task("[cyan]Processing tickets...", total=len(df))

            for idx, row in df.iterrows():
                result = self._process_row(row)
                results.append(result)

                if self.verbose:
                    self._print_result(idx, result)

                progress.advance(task)
                time.sleep(0.1)

        out_df = pd.DataFrame(results)
        out_df.to_csv(output_path, index=False)

        console.print(f"\n[green]Saved results → {output_path}[/green]")
        self._print_summary(out_df)

    # ─────────────────────────────────────────────────────────
    def _process_row(self, row: pd.Series) -> dict:
        issue = str(row.get("issue", "")).strip()
        subject = str(row.get("subject", "")).strip()
        company = str(row.get("company", "")).strip()

        text = self._normalize(issue, subject)

        if not text:
            return self._fallback()

        # Risk check
        risk = self.risk_checker.check(text)

        # Company inference
        if not company:
            company = self._infer_company(text)

        # Retrieval
        docs = self.retriever.search(text, company, RETRIEVAL_TOP_K)
        corpus_found = bool(docs) and docs[0]["score"] >= MIN_CORPUS_SCORE

        # Decision logic
        if risk["must_escalate"]:
            result = {
                "status": "escalated",
                "product_area": "risk",
                "response": "This issue requires human support.",
                "justification": f"High risk detected ({risk['level']})",
                "request_type": "product_issue",
            }

        elif corpus_found:
            top_doc = docs[0]["text"][:200]

            result = {
                "status": "replied",
                "product_area": company.lower(),
                "response": f"Based on support docs: {top_doc}",
                "justification": "Response grounded in support corpus",
                "request_type": "product_issue",
            }

        else:
            result = {
                "status": "replied",
                "product_area": "general",
                "response": "Please check the help center or contact support.",
                "justification": "No strong corpus match but safe to respond",
                "request_type": "product_issue",
            }

        return self._validate(result)

    # ─────────────────────────────────────────────────────────
    @staticmethod
    def _normalize(issue: str, subject: str) -> str:
        parts = []
        if subject:
            parts.append(subject)
        if issue:
            parts.append(issue)

        text = " ".join(parts)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    # ─────────────────────────────────────────────────────────
    def _infer_company(self, text: str) -> str:
        t = text.lower()

        if any(k in t for k in ["payment", "card", "transaction", "bank"]):
            return "Visa"
        if any(k in t for k in ["claude", "ai", "model"]):
            return "Claude"
        if any(k in t for k in ["test", "coding", "hackerrank"]):
            return "HackerRank"

        return "general"

    # ─────────────────────────────────────────────────────────
    @staticmethod
    def _fallback():
        return {
            "status": "escalated",
            "product_area": "unknown",
            "response": "Unable to process request.",
            "justification": "Empty or invalid input",
            "request_type": "invalid",
        }

    # ─────────────────────────────────────────────────────────
    @staticmethod
    def _validate(result: dict) -> dict:
        if result["status"] not in ALLOWED_STATUS:
            result["status"] = "escalated"

        if result["request_type"] not in ALLOWED_REQUEST_TYPE:
            result["request_type"] = "product_issue"

        return result

    # ─────────────────────────────────────────────────────────
    def _print_result(self, idx: int, result: dict):
        color = "red" if result["status"] == "escalated" else "green"
        console.print(f"[{color}]#{idx+1} {result['status']}[/{color}]")

    # ─────────────────────────────────────────────────────────
    @staticmethod
    def _print_summary(df: pd.DataFrame):
        table = Table(title="Summary")
        table.add_column("Metric")
        table.add_column("Count")

        table.add_row("Total", str(len(df)))
        table.add_row("Replied", str((df["status"] == "replied").sum()))
        table.add_row("Escalated", str((df["status"] == "escalated").sum()))

        console.print(table)