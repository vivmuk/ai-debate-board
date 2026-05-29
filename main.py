#!/usr/bin/env python3
"""
AI Debate Board — Main Entry Point

Run multi-agent AI debates on strategic decisions using Venice.ai APIs.

Usage:
  # Run a debate
  python3 main.py --scenario scenarios/fomc_june_2026.yaml
  python3 main.py --scenario scenarios/fomc_june_2026.yaml --dual-model
  python3 main.py --scenarios-dir scenarios/ --dual-model

  # Record a forward-looking prediction
  python3 main.py --predict --scenario scenarios/fomc_june_2026.yaml

  # Score a prediction after the outcome is known
  python3 main.py --score PREDICTION_ID --outcome hold

  # View prediction ledger
  python3 main.py --ledger
"""

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

from venice_client import VeniceConfig
from parallel_engine import ParallelDebateEngine
from debate_engine import (
    DualModelResult,
    DebateResult,
    load_scenario,
    format_debate_summary,
    format_dual_model_summary,
    save_result,
)
from prediction_tracker import PredictionTracker, run_and_record_prediction, score_existing_prediction


console = Console()


def run_single_scenario(scenario_path: str, engine,
                        dual_model: bool, model: str = None,
                        output_dir: str = "output") -> None:
    """Run a single scenario through the debate engine."""
    scenario = load_scenario(scenario_path)

    console.print(Panel(
        f"[bold]{scenario['title']}[/bold]\n"
        f"Date: {scenario['decision_date']} | Category: {scenario.get('category', 'general')}",
        title="📋 Scenario Loaded",
        border_style="blue",
    ))

    if dual_model:
        result = engine.run_dual_model_debate(scenario)
        console.print("\n" + format_dual_model_summary(result))
        save_result(result, output_dir)
    else:
        result = engine.run_debate(scenario, model=model)
        console.print("\n" + format_debate_summary(result))
        save_result(result, output_dir)


def run_scenarios_dir(scenarios_dir: str, engine,
                      dual_model: bool, output_dir: str = "output") -> None:
    """Run all scenarios in a directory."""
    scenarios_path = Path(scenarios_dir)
    yaml_files = sorted(scenarios_path.glob("*.yaml")) + sorted(scenarios_path.glob("*.yml"))

    if not yaml_files:
        console.print(f"[red]No scenario files found in {scenarios_dir}[/red]")
        return

    console.print(f"\n📂 Found {len(yaml_files)} scenarios in {scenarios_dir}")

    results_summary = []
    for yaml_file in yaml_files:
        scenario = load_scenario(str(yaml_file))

        if dual_model:
            result = engine.run_dual_model_debate(scenario)
            save_result(result, output_dir)
            results_summary.append({
                "scenario": scenario["scenario_id"],
                "model_a": result.model_a_result.model,
                "model_a_decision": result.model_a_result.decision_label,
                "model_a_prob": result.model_a_result.probability,
                "model_b": result.model_b_result.model,
                "model_b_decision": result.model_b_result.decision_label,
                "model_b_prob": result.model_b_result.probability,
                "agreement": result.agreement,
            })
        else:
            result = engine.run_debate(scenario)
            save_result(result, output_dir)
            results_summary.append({
                "scenario": scenario["scenario_id"],
                "model": result.model,
                "decision": result.decision_label,
                "probability": result.probability,
                "confidence": result.confidence,
            })

    # Print summary table
    console.print("\n")
    console.print(Panel("📊 ALL SCENARIOS SUMMARY", border_style="green"))

    if dual_model:
        table = Table(title="Dual-Model Cross-Validation Results")
        table.add_column("Scenario", style="cyan")
        table.add_column("Model A Decision", style="magenta")
        table.add_column("Model A Prob", justify="right")
        table.add_column("Model B Decision", style="yellow")
        table.add_column("Model B Prob", justify="right")
        table.add_column("Agreement", justify="center")

        for r in results_summary:
            agree_str = "✅" if r["agreement"] else "⚠️"
            table.add_row(
                r["scenario"],
                r["model_a_decision"],
                f"{r['model_a_prob']}%",
                r["model_b_decision"],
                f"{r['model_b_prob']}%",
                agree_str,
            )
    else:
        table = Table(title="Debate Results")
        table.add_column("Scenario", style="cyan")
        table.add_column("Model", style="magenta")
        table.add_column("Decision", style="yellow")
        table.add_column("Probability", justify="right")
        table.add_column("Confidence", justify="center")

        for r in results_summary:
            table.add_row(
                r["scenario"],
                r["model"],
                r["decision"],
                f"{r['probability']}%",
                r["confidence"],
            )

    console.print(table)


def show_ledger(ledger_path: str = "predictions_ledger.json"):
    """Display the prediction ledger with rich formatting."""
    tracker = PredictionTracker(ledger_path)
    
    console.print(Panel(
        f"Total: {len(tracker.predictions)} | "
        f"Pending: {len(tracker.list_pending())} | "
        f"Scored: {len(tracker.list_scored())}",
        title="📊 Prediction Ledger",
        border_style="bright_blue",
    ))
    
    # Pending predictions
    pending = tracker.list_pending()
    if pending:
        console.print("\n[bold yellow]⏳ Pending Predictions[/bold yellow]")
        table = Table(show_header=True)
        table.add_column("ID", style="dim")
        table.add_column("Question", style="cyan")
        table.add_column("Model A Decision", style="magenta")
        table.add_column("Model A Prob", justify="right")
        table.add_column("Model B Decision", style="yellow")
        table.add_column("Model B Prob", justify="right")
        table.add_column("Decision Date", style="green")
        
        for p in pending:
            table.add_row(
                p.prediction_id[:40],
                p.question[:50],
                p.model_a_decision,
                f"{p.model_a_probability}%",
                p.model_b_decision or "—",
                f"{p.model_b_probability}%" if p.model_b else "—",
                p.decision_date,
            )
        console.print(table)
    
    # Scored predictions
    scored = tracker.list_scored()
    if scored:
        console.print("\n[bold green]✅ Scored Predictions[/bold green]")
        table = Table(show_header=True)
        table.add_column("Question", style="cyan")
        table.add_column("Predicted", style="magenta")
        table.add_column("Prob", justify="right")
        table.add_column("Actual", style="yellow")
        table.add_column("Result", justify="center")
        table.add_column("Brier", justify="right")
        
        for p in scored:
            icon = "✅" if p.correct else "❌"
            table.add_row(
                p.question[:40],
                p.model_a_decision,
                f"{p.model_a_probability}%",
                p.actual_outcome,
                icon,
                f"{p.brier_score_a:.3f}",
            )
        console.print(table)
    
    # Calibration
    cal = tracker.get_calibration_data()
    if cal.get("scored_count", 0) > 0:
        console.print("\n[bold blue]📈 Calibration Summary[/bold blue]")
        console.print(f"  Accuracy: {cal['accuracy']}%")
        console.print(f"  Avg Brier Score (A): {cal['avg_brier_score_a']:.3f} (0=perfect, 1=worst)")
        if cal.get("avg_brier_score_b") is not None:
            console.print(f"  Avg Brier Score (B): {cal['avg_brier_score_b']:.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="AI Debate Board — Multi-agent strategic decision-making via Venice.ai",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a debate
  %(prog)s --scenario scenarios/fomc_june_2026.yaml
  %(prog)s --scenario scenarios/fomc_june_2026.yaml --dual-model

  # Record as a prediction (forward-looking scenarios only)
  %(prog)s --predict --scenario scenarios/fomc_june_2026.yaml

  # Score a prediction
  %(prog)s --score PREDICTION_ID --outcome hold

  # View prediction ledger
  %(prog)s --ledger
        """,
    )

    # Action modes
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Run debate AND record as a tracked prediction",
    )
    parser.add_argument(
        "--score",
        metavar="PREDICTION_ID",
        help="Score a prediction with the actual outcome (use --outcome)",
    )
    parser.add_argument(
        "--outcome",
        help="The actual outcome option ID (used with --score)",
    )
    parser.add_argument(
        "--ledger",
        action="store_true",
        help="View the prediction ledger",
    )

    # Scenario options
    parser.add_argument(
        "--scenario", "-s",
        help="Path to a single scenario YAML file",
    )
    parser.add_argument(
        "--scenarios-dir", "-d",
        help="Path to directory of scenario YAML files",
    )
    parser.add_argument(
        "--dual-model",
        action="store_true",
        help="Run scenario on two models and compare (cross-validation)",
    )
    parser.add_argument(
        "--model", "-m",
        help="Override the primary model (default: from config)",
    )
    parser.add_argument(
        "--model-secondary",
        help="Override the secondary model for dual-model runs",
    )
    parser.add_argument(
        "--rounds", "-r",
        type=int,
        default=3,
        help="Number of debate rounds (default: 3)",
    )
    parser.add_argument(
        "--output", "-o",
        default="output",
        help="Output directory for results (default: output/)",
    )

    args = parser.parse_args()

    # ─── Ledger view mode ────────────────────────────────
    if args.ledger:
        show_ledger()
        return

    # ─── Score mode ──────────────────────────────────────
    if args.score:
        if not args.outcome:
            parser.error("--outcome is required when using --score")
        score_existing_prediction(args.score, args.outcome)
        return

    # ─── Debate / Predict mode ───────────────────────────
    if not args.scenario and not args.scenarios_dir:
        parser.error("Provide --scenario or --scenarios-dir (or use --ledger)")

    # Load Venice config
    config = VeniceConfig.from_env_and_hermes()
    if args.model:
        config.primary_model = args.model
    if args.model_secondary:
        config.secondary_model = args.model_secondary

    mode_label = "🎯 PREDICTION" if args.predict else "🎭 DEBATE"

    console.print(Panel(
        f"Primary Model: [bold]{config.primary_model}[/bold]\n"
        f"Secondary Model: [bold]{config.secondary_model}[/bold]\n"
        f"Dual-Model: {'✅' if args.dual_model else '❌'}\n"
        f"Debate Rounds: {args.rounds}\n"
        f"Mode: {mode_label}",
        title=f"{mode_label} Board",
        border_style="bright_blue",
    ))

    # ─── Prediction mode ────────────────────────────────
    if args.predict and args.scenario:
        run_and_record_prediction(
            args.scenario,
            dual_model=args.dual_model,
            output_dir=args.output,
        )
        return

    # ─── Standard debate mode ───────────────────────────
    engine = ParallelDebateEngine(config, max_workers=5)

    if args.scenario:
        run_single_scenario(
            args.scenario, engine, args.dual_model, args.model, args.output
        )
    elif args.scenarios_dir:
        run_scenarios_dir(args.scenarios_dir, engine, args.dual_model, args.output)


if __name__ == "__main__":
    main()
