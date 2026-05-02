#!/usr/bin/env python3
"""
Multi-Domain Support Triage Agent
Terminal Entry Point — Hackathon Ready
"""

import argparse
import sys
import os
from pathlib import Path

# Make src/ importable
sys.path.insert(0, str(Path(__file__).parent))

from src.agent import TriageAgent
from src.corpus_loader import CorpusLoader
from rich.console import Console
from rich.panel import Panel

console = Console()

BANNER = """
╔══════════════════════════════════════════════════════╗
║        MULTI-DOMAIN SUPPORT TRIAGE AGENT             ║
║   HackerRank · Claude · Visa                        ║
╚══════════════════════════════════════════════════════╝
"""


def main():
    parser = argparse.ArgumentParser(
        description="Terminal-based AI support triage agent"
    )

    parser.add_argument(
        "--input",
        default="data/support_issues.csv",
        help="Path to input CSV",
    )

    parser.add_argument(
        "--output",
        default="output/results.csv",
        help="Path for output CSV",
    )

    parser.add_argument(
        "--corpus",
        default="data/corpus",
        help="Directory containing support corpus",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed logs",
    )

    args = parser.parse_args()

    console.print(BANNER, style="bold cyan")

    # ── Validate input ─────────────────────────────────────────
    if not os.path.exists(args.input):
        console.print(f"[red]ERROR:[/red] Input file not found: {args.input}")
        sys.exit(1)

    # Ensure output directory exists
    os.makedirs(Path(args.output).parent, exist_ok=True)

    # ── Load corpus ────────────────────────────────────────────
    console.print(f"[bold]Loading corpus from:[/bold] {args.corpus}")
    loader = CorpusLoader(args.corpus)
    corpus = loader.load()
    console.print(f"[green]✓ Loaded {len(corpus)} corpus chunks[/green]")

    # ── Run agent ──────────────────────────────────────────────
    console.print(f"\n[bold]Processing:[/bold] {args.input}")

    agent = TriageAgent(corpus=corpus, verbose=args.verbose)
    agent.run(input_path=args.input, output_path=args.output)

    # ── Done ───────────────────────────────────────────────────
    console.print(
        Panel(
            f"[bold green]✓ Done![/bold green]\nResults saved to: [cyan]{args.output}[/cyan]",
            title="Complete",
        )
    )


if __name__ == "__main__":
    main()