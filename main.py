#!/usr/bin/env python3
"""
Multi-Domain Support Triage Agent
Orchestrate May'26 Hackathon — Terminal Entry Point
"""

import argparse
import sys
import os
from pathlib import Path

# ── make sure src/ is importable ──────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from src.agent import TriageAgent
from src.corpus_loader import CorpusLoader
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()

BANNER = """
╔══════════════════════════════════════════════════════╗
║        MULTI-DOMAIN SUPPORT TRIAGE AGENT             ║
║   HackerRank · Claude · Visa  — Orchestrate May'26   ║
╚══════════════════════════════════════════════════════╝
"""


def main():
    parser = argparse.ArgumentParser(
        description="Terminal-based AI support triage agent"
    )
    parser.add_argument(
        "--input",
        default="data/support_issues.csv",
        help="Path to input CSV (default: data/support_issues.csv)",
    )
    parser.add_argument(
        "--output",
        default="output/results.csv",
        help="Path for output CSV (default: output/results.csv)",
    )
    parser.add_argument(
        "--corpus",
        default="data/corpus",
        help="Directory containing support corpus text files",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Run on sample_support_issues.csv for validation",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed reasoning for each ticket",
    )
    parser.add_argument(
        "--scrape-corpus",
        action="store_true",
        help="Re-scrape support corpus from live URLs before running",
    )
    args = parser.parse_args()

    console.print(BANNER, style="bold cyan")

    # ── optionally scrape corpus ──────────────────────────────────────────────
    if args.scrape_corpus:
        console.print("[yellow]Scraping support corpus from live URLs...[/yellow]")
        from src.corpus_loader import scrape_all_corpus
        scrape_all_corpus(args.corpus)

    # ── resolve paths ─────────────────────────────────────────────────────────
    input_path = "data/sample_support_issues.csv" if args.sample else args.input
    output_path = args.output

    if not os.path.exists(input_path):
        console.print(f"[red]ERROR:[/red] Input file not found: {input_path}")
        sys.exit(1)

    os.makedirs(Path(output_path).parent, exist_ok=True)

    # ── load corpus ───────────────────────────────────────────────────────────
    console.print(f"[bold]Loading corpus from:[/bold] {args.corpus}")
    loader = CorpusLoader(args.corpus)
    corpus = loader.load()
    console.print(f"[green]✓ Loaded {len(corpus)} corpus chunks[/green]")

    # ── run agent ─────────────────────────────────────────────────────────────
    console.print(f"\n[bold]Processing:[/bold] {input_path}")
    agent = TriageAgent(corpus=corpus, verbose=args.verbose)
    agent.run(input_path=input_path, output_path=output_path)

    console.print(
        Panel(
            f"[bold green]✓ Done![/bold green]\nResults saved to: [cyan]{output_path}[/cyan]",
            title="Complete",
        )
    )


if __name__ == "__main__":
    main()