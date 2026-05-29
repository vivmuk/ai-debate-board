#!/usr/bin/env python3
"""
AI Debate Board — Main Entry Point

Run multi-agent AI debates on strategic decisions using Venice.ai APIs.

Usage:
  python main.py --scenario scenarios/bms_checkmate026.yaml
  python main.py --scenario scenarios/bms_checkmate026.yaml --dual-model
  python main.py --scenarios-dir scenarios/ --dual-model
  python main.py --scenario scenarios/bms_checkmate026.yaml --model grok-4-20
  python main.py --scenario scenarios/bms_checkmate026.yaml --rounds 5
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


console = Console()


def run_single_scenario(scenario_path: str, engine: DebateEngine,
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


def run_scenarios_dir(scenarios_dir: str, engine: DebateEngine,
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


def main():
    parser = argparse.ArgumentParser(
        description="AI Debate Board — Multi-agent strategic decision-making via Venice.ai",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --scenario scenarios/bms_checkmate026.yaml
  %(prog)s --scenario scenarios/bms_checkmate026.yaml --dual-model
  %(prog)s --scenarios-dir scenarios/ --dual-model --output results/
        """,
    )
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

    if not args.scenario and not args.scenarios_dir:
        parser.error("Provide --scenario or --scenarios-dir")

    # Load Venice config
    config = VeniceConfig.from_env_and_hermes()
    if args.model:
        config.primary_model = args.model
    if args.model_secondary:
        config.secondary_model = args.model_secondary

    console.print(Panel(
        f"Primary Model: [bold]{config.primary_model}[/bold]\n"
        f"Secondary Model: [bold]{config.secondary_model}[/bold]\n"
        f"Dual-Model: {'✅' if args.dual_model else '❌'}\n"
        f"Debate Rounds: {args.rounds}",
        title="🎭 AI Debate Board",
        border_style="bright_blue",
    ))

    # Create engine (parallel execution)
    engine = ParallelDebateEngine(config, max_workers=5)

    # Run
    if args.scenario:
        # Override rounds in scenario if specified
        run_single_scenario(
            args.scenario, engine, args.dual_model, args.model, args.output
        )
    elif args.scenarios_dir:
        run_scenarios_dir(args.scenarios_dir, engine, args.dual_model, args.output)


if __name__ == "__main__":
    main()
