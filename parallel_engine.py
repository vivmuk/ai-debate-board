"""
AI Debate Board — Parallel Debate Engine

Runs agent turns concurrently using ThreadPoolExecutor for dramatically
faster debate execution. A 5-agent × 3-round debate that took ~8 minutes
sequentially now runs in ~2-3 minutes.
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from dataclasses import dataclass, field

from venice_client import VeniceConfig, venice_chat, extract_content, extract_usage
from debate_engine import (
    DebateResult, DualModelResult, AgentMessage,
    load_scenario, build_agent_system_prompt, build_debate_round_prompt,
    build_ceo_synthesis_prompt, format_debate_summary, format_dual_model_summary,
    save_result,
)


class ParallelDebateEngine:
    """Orchestrates multi-agent debates with parallel agent execution."""

    def __init__(self, config: VeniceConfig, max_workers: int = 5):
        self.config = config
        self.max_workers = max_workers
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def _run_single_agent_turn(
        self,
        role: dict,
        scenario: dict,
        round_num: int,
        total_rounds: int,
        previous_messages: list,
        model: str,
    ) -> AgentMessage:
        """Run a single agent's turn — designed for parallel execution."""
        system_prompt = build_agent_system_prompt(role, scenario)
        user_prompt = build_debate_round_prompt(
            role, round_num, total_rounds, previous_messages, scenario,
            steel_man=scenario.get("debate_config", {}).get("steel_man_required", True),
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = venice_chat(
            self.config,
            messages=messages,
            model=model,
            temperature=0.7,
            max_completion_tokens=1500,
        )

        content = extract_content(response)
        usage = extract_usage(response)

        return AgentMessage(
            role_id=role["id"],
            role_title=role["title"],
            round_num=round_num,
            content=content,
        ), usage

    def _run_ceo_synthesis(self, scenario: dict, all_messages: list, model: str) -> dict:
        """Run CEO synthesis to produce final calibrated decision."""
        system_prompt = (
            "You are the CEO of the executive team. Your job is to make the FINAL "
            "strategic decision after hearing all arguments from your leadership team. "
            "You must be decisive, weigh competing perspectives, and commit to a single "
            "course of action with a calibrated probability estimate.\n\n"
            "You must respond with valid JSON only. No markdown, no commentary outside the JSON."
        )

        user_prompt = build_ceo_synthesis_prompt(scenario, all_messages)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = venice_chat(
            self.config,
            messages=messages,
            model=model,
            temperature=0.5,
            max_completion_tokens=1500,
            response_format={"type": "json_object"},
        )

        content = extract_content(response)
        usage = extract_usage(response)
        self.total_input_tokens += usage.get("prompt_tokens", 0)
        self.total_output_tokens += usage.get("completion_tokens", 0)

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            return {
                "decision_id": "parse_error",
                "decision_label": "Could not parse CEO synthesis",
                "probability": 0,
                "confidence": "unknown",
                "reasoning": content[:500],
                "key_risks": [],
                "overruled_positions": [],
            }

    def run_debate(
        self,
        scenario: dict,
        model: Optional[str] = None,
    ) -> DebateResult:
        """Run a complete debate with parallel agent execution within each round."""
        model = model or self.config.primary_model
        debate_config = scenario.get("debate_config", {})
        total_rounds = debate_config.get("rounds", 2)
        roles = scenario.get("roles", [])

        # Filter out CEO — they only come in for synthesis
        debate_roles = [r for r in roles if not r.get("synthesis_role")]

        print(f"\n{'='*60}")
        print(f"🎭 DEBATE: {scenario['title']}")
        print(f"📋 Model: {model} | Rounds: {total_rounds} | Agents: {len(debate_roles)}")
        print(f"⚡ Parallel workers: {self.max_workers}")
        print(f"{'='*60}")

        all_messages: list[AgentMessage] = []

        # Run debate rounds — agents within a round run in PARALLEL
        for round_num in range(1, total_rounds + 1):
            print(f"\n--- Round {round_num}/{total_rounds} (parallel) ---")

            # Submit all agent turns for this round in parallel
            futures = {}
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                for role in debate_roles:
                    future = executor.submit(
                        self._run_single_agent_turn,
                        role, scenario, round_num, total_rounds,
                        all_messages, model,
                    )
                    futures[future] = role

                # Collect results as they complete
                for future in as_completed(futures):
                    role = futures[future]
                    try:
                        msg, usage = future.result()
                        all_messages.append(msg)
                        self.total_input_tokens += usage.get("prompt_tokens", 0)
                        self.total_output_tokens += usage.get("completion_tokens", 0)

                        pos = "unknown"
                        for opt in scenario["options"]:
                            if opt["id"] in msg.content.lower():
                                pos = opt["label"]
                                break
                        print(f"  ✅ {role['title']} → {pos}")
                    except Exception as e:
                        print(f"  ❌ {role['title']} failed: {e}")

        # CEO synthesis (single call, no parallelism needed)
        print(f"\n  👑 CEO synthesizing final decision...", end=" ", flush=True)
        decision = self._run_ceo_synthesis(scenario, all_messages, model)
        print(f"→ {decision.get('decision_label', 'unknown')} "
              f"({decision.get('probability', '?')}%)")

        # Build agent arguments summary
        agent_arguments = {}
        for role in debate_roles:
            role_msgs = [m for m in all_messages if m.role_id == role["id"]]
            if role_msgs:
                agent_arguments[role["id"]] = role_msgs[-1].content

        return DebateResult(
            scenario_id=scenario["scenario_id"],
            model=model,
            decision=decision.get("decision_id", "unknown"),
            decision_label=decision.get("decision_label", "unknown"),
            probability=decision.get("probability", 0),
            confidence=decision.get("confidence", "unknown"),
            reasoning=decision.get("reasoning", ""),
            agent_arguments=agent_arguments,
            debate_log=all_messages,
            ceo_synthesis=json.dumps(decision, indent=2),
            token_usage={
                "input_tokens": self.total_input_tokens,
                "output_tokens": self.total_output_tokens,
            },
        )

    def run_dual_model_debate(
        self,
        scenario: dict,
        model_a: Optional[str] = None,
        model_b: Optional[str] = None,
    ) -> DualModelResult:
        """Run the same scenario on two models and compare results."""
        model_a = model_a or self.config.primary_model
        model_b = model_b or self.config.secondary_model

        print(f"\n{'#'*60}")
        print(f"⚖️  DUAL-MODEL DEBATE: {scenario['title']}")
        print(f"   Model A: {model_a}")
        print(f"   Model B: {model_b}")
        print(f"{'#'*60}")

        # Model A run
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        result_a = self.run_debate(scenario, model=model_a)

        # Model B run
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        result_b = self.run_debate(scenario, model=model_b)

        # Compare
        agreement = result_a.decision == result_b.decision
        flags = []

        if not agreement:
            flags.append(
                f"DECISION DIVERGENCE: {model_a} chose '{result_a.decision_label}' "
                f"vs {model_b} chose '{result_b.decision_label}' — "
                f"FLAG FOR HUMAN REVIEW"
            )

        prob_diff = abs(result_a.probability - result_b.probability)
        if prob_diff > 20:
            flags.append(
                f"PROBABILITY GAP: {prob_diff:.0f}% difference "
                f"({result_a.probability}% vs {result_b.probability}%)"
            )

        if result_a.confidence != result_b.confidence:
            flags.append(
                f"CONFIDENCE MISMATCH: {model_a}={result_a.confidence} "
                f"vs {model_b}={result_b.confidence}"
            )

        return DualModelResult(
            scenario_id=scenario["scenario_id"],
            model_a_result=result_a,
            model_b_result=result_b,
            agreement=agreement,
            disagreement_flags=flags,
        )
