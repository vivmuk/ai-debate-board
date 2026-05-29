"""
AI Debate Board — Debate Engine

Core orchestration: loads scenarios, spawns agents, runs multi-round debate,
and synthesizes final decisions with calibrated probabilities.
"""

import json
import uuid
import yaml
from datetime import datetime
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

from venice_client import VeniceConfig, venice_chat, extract_content, extract_usage


# ─── Data structures ────────────────────────────────────────────────

@dataclass
class AgentMessage:
    role_id: str
    role_title: str
    round_num: int
    content: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class DebateResult:
    scenario_id: str
    model: str
    decision: str
    decision_label: str
    probability: float
    confidence: str
    reasoning: str
    agent_arguments: dict  # role_id -> final argument
    debate_log: list  # all messages in order
    ceo_synthesis: str
    token_usage: dict
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class DualModelResult:
    scenario_id: str
    model_a_result: DebateResult
    model_b_result: DebateResult
    agreement: bool
    disagreement_flags: list


# ─── Scenario loader ────────────────────────────────────────────────

def load_scenario(path: str) -> dict:
    """Load a scenario YAML file."""
    with open(path) as f:
        return yaml.safe_load(f)


# ─── Prompt builders ───────────────────────────────────────────────

def build_agent_system_prompt(role: dict, scenario: dict) -> str:
    """Build the system prompt for a specific agent role."""
    options_text = "\n".join(
        f"  {i+1}. **{opt['label']}** (ID: {opt['id']}): {opt['description']}"
        for i, opt in enumerate(scenario["options"])
    )

    evidence_text = "\n\n".join(
        f"### {ev['source']} ({ev['date']})\n{ev['content']}"
        for ev in scenario.get("evidence", [])
    )

    constraints_text = "\n".join(
        f"- {c}" for c in scenario.get("constitutional_constraints", [])
    )

    return f"""You are the **{role['title']}** of the executive leadership team making a critical strategic decision.

## Your Mandate
{role['mandate']}

## The Decision
{scenario['title']}

Decision Date: {scenario['decision_date']}

{scenario['context']}

## Available Options
{options_text}

## Available Evidence (time-locked to {scenario['decision_date']})
{evidence_text}

## Constitutional Constraints (YOU MUST OBEY THESE)
{constraints_text}

## Debate Rules
1. Argue from YOUR role's perspective and incentives — not as a generic advisor.
2. If steel-man mode is enabled, you must first accurately restate the strongest version of any opposing argument before rebutting it.
3. You may NOT use any information dated after {scenario['decision_date']}.
4. Be specific: cite evidence, quantify risks, estimate probabilities where possible.
5. End each argument with your current position (which option ID you favor) and a brief justification.

## Output Format
For each argument, end with:
- POSITION: [option_id]
- CONFIDENCE: [low/medium/high]
- KEY REASON: [one sentence]
"""


def build_debate_round_prompt(
    role: dict,
    round_num: int,
    total_rounds: int,
    previous_messages: list[AgentMessage],
    scenario: dict,
    steel_man: bool = True,
) -> str:
    """Build the user prompt for a specific debate round."""
    # Format previous round messages
    prior_text = ""
    if previous_messages:
        prior_text = "## Arguments from Previous Rounds\n\n"
        for msg in previous_messages:
            prior_text += f"**{msg.role_title}** (Round {msg.round_num}):\n{msg.content}\n\n---\n\n"

    if round_num == 1:
        instruction = """This is the opening round. Present your initial position based on your role's mandate.
Argue from your specific perspective — what does your role care about most?
Be specific about which option you favor and why."""
    elif round_num == total_rounds:
        instruction = f"""This is the FINAL round. You've heard all arguments.
You may revise your position based on the debate. Give your definitive argument.
You must commit to a single option with a calibrated probability (0-100%) that your chosen option will prove correct."""
    else:
        instruction = f"""This is round {round_num} of {total_rounds}.
Respond to the strongest points from other roles. {"Apply steel-man: restate the strongest opposing argument before rebutting." if steel_man else ""}
You may revise your position if persuaded by new arguments."""

    return f"""{prior_text}

## Your Task — Round {round_num} of {total_rounds}

{instruction}

Remember: You are the {role['title']}. Argue from YOUR perspective."""


def build_ceo_synthesis_prompt(scenario: dict, all_messages: list[AgentMessage]) -> str:
    """Build the prompt for the CEO to synthesize all arguments and make a final decision."""
    debate_text = ""
    for msg in all_messages:
        debate_text += f"### {msg.role_title} (Round {msg.round_num})\n{msg.content}\n\n"

    options_text = "\n".join(
        f"  - {opt['id']}: {opt['label']}"
        for opt in scenario["options"]
    )

    return f"""As CEO, you have now heard all arguments from your executive team across multiple rounds of debate.

## Complete Debate Record

{debate_text}

## Your Task as CEO

Synthesize all arguments. Weigh the evidence, the risks, and the opportunities.
Make a FINAL DECISION on behalf of the executive team.

You must:
1. Acknowledge the strongest argument for each option
2. Explain why you are overriding or accepting each team member's position
3. Commit to a SINGLE option
4. Assign a calibrated probability (0-100%) that your chosen option will prove to be the correct decision
5. State your confidence level

## Available Options
{options_text}

## Constitutional Constraints
You may NOT use any information dated after {scenario['decision_date']}.

## Output Format (JSON)
Respond with a JSON object:
```json
{{
  "decision_id": "option_id_here",
  "decision_label": "Full label of chosen option",
  "probability": 65,
  "confidence": "medium",
  "reasoning": "Your detailed synthesis of why this is the right call...",
  "key_risks": ["Risk 1", "Risk 2"],
  "overruled_positions": ["Which agent positions you're overriding and why"]
}}
```"""


# ─── Debate execution ──────────────────────────────────────────────

class DebateEngine:
    """Orchestrates multi-agent debates via Venice API."""

    def __init__(self, config: VeniceConfig):
        self.config = config
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def run_agent_turn(
        self,
        role: dict,
        scenario: dict,
        round_num: int,
        total_rounds: int,
        previous_messages: list[AgentMessage],
        model: str,
    ) -> AgentMessage:
        """Run a single agent's turn in the debate."""
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
            max_completion_tokens=2048,
        )

        content = extract_content(response)
        usage = extract_usage(response)
        self.total_input_tokens += usage.get("prompt_tokens", 0)
        self.total_output_tokens += usage.get("completion_tokens", 0)

        return AgentMessage(
            role_id=role["id"],
            role_title=role["title"],
            round_num=round_num,
            content=content,
        )

    def run_ceo_synthesis(
        self,
        scenario: dict,
        all_messages: list[AgentMessage],
        model: str,
    ) -> dict:
        """Run the CEO synthesis to produce the final decision."""
        system_prompt = f"""You are the CEO of the executive team. Your job is to make the FINAL strategic decision after hearing all arguments from your leadership team. You must be decisive, weigh competing perspectives, and commit to a single course of action with a calibrated probability estimate.

You must respond with valid JSON only. No markdown, no commentary outside the JSON."""

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
            max_completion_tokens=2048,
            response_format={"type": "json_object"},
        )

        content = extract_content(response)
        usage = extract_usage(response)
        self.total_input_tokens += usage.get("prompt_tokens", 0)
        self.total_output_tokens += usage.get("completion_tokens", 0)

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            return {
                "decision_id": "parse_error",
                "decision_label": "Could not parse CEO synthesis",
                "probability": 0,
                "confidence": "unknown",
                "reasoning": content,
                "key_risks": [],
                "overruled_positions": [],
            }

    def run_debate(
        self,
        scenario: dict,
        model: Optional[str] = None,
    ) -> DebateResult:
        """Run a complete debate for a scenario on a single model."""
        model = model or self.config.primary_model
        debate_config = scenario.get("debate_config", {})
        total_rounds = debate_config.get("rounds", 3)
        roles = scenario.get("roles", [])

        # Filter out CEO — they only come in for synthesis
        debate_roles = [r for r in roles if not r.get("synthesis_role")]
        ceo_role = next((r for r in roles if r.get("synthesis_role")), None)

        print(f"\n{'='*60}")
        print(f"🎭 DEBATE: {scenario['title']}")
        print(f"📋 Model: {model} | Rounds: {total_rounds} | Agents: {len(debate_roles)}")
        print(f"{'='*60}")

        all_messages: list[AgentMessage] = []

        # Run debate rounds
        for round_num in range(1, total_rounds + 1):
            print(f"\n--- Round {round_num}/{total_rounds} ---")

            for role in debate_roles:
                print(f"  🗣️  {role['title']} is arguing...", end=" ", flush=True)
                msg = self.run_agent_turn(
                    role=role,
                    scenario=scenario,
                    round_num=round_num,
                    total_rounds=total_rounds,
                    previous_messages=all_messages,
                    model=model,
                )
                all_messages.append(msg)

                # Quick parse of their position
                pos = "unknown"
                for opt in scenario["options"]:
                    if opt["id"] in msg.content.lower():
                        pos = opt["label"]
                        break
                print(f"→ {pos}")

        # CEO synthesis
        print(f"\n  👑 CEO synthesizing final decision...", end=" ", flush=True)
        decision = self.run_ceo_synthesis(scenario, all_messages, model)
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

        # Reset token counters for model A run
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        result_a = self.run_debate(scenario, model=model_a)

        # Reset for model B run
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        result_b = self.run_debate(scenario, model=model_b)

        # Compare results
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
                f"({result_a.probability}% vs {result_b.probability}%) — "
                f"significant calibration disagreement"
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


# ─── Output formatters ──────────────────────────────────────────────

def format_debate_summary(result: DebateResult) -> str:
    """Format a single-model debate result for display."""
    lines = [
        f"📊 DEBATE RESULT: {result.scenario_id}",
        f"   Model: {result.model}",
        f"   Decision: {result.decision_label} (ID: {result.decision})",
        f"   Probability: {result.probability}%",
        f"   Confidence: {result.confidence}",
        f"   Tokens: {result.token_usage.get('input_tokens', 0)} in / "
        f"{result.token_usage.get('output_tokens', 0)} out",
        "",
        "📝 CEO Reasoning:",
        f"   {result.reasoning[:500]}...",
    ]
    return "\n".join(lines)


def format_dual_model_summary(result: DualModelResult) -> str:
    """Format a dual-model comparison for display."""
    a, b = result.model_a_result, result.model_b_result

    lines = [
        f"⚖️  DUAL-MODEL COMPARISON: {result.scenario_id}",
        f"",
        f"   {'':15} {'Model A':25} {'Model B':25}",
        f"   {'─'*15} {'─'*25} {'─'*25}",
        f"   {'Decision':15} {a.decision_label:25} {b.decision_label:25}",
        f"   {'Probability':15} {a.probability:>24}% {b.probability:>24}%",
        f"   {'Confidence':15} {a.confidence:25} {b.confidence:25}",
        f"",
        f"   Agreement: {'✅ YES' if result.agreement else '⚠️ NO — FLAG FOR HUMAN REVIEW'}",
    ]

    if result.disagreement_flags:
        lines.append("")
        lines.append("🚩 Disagreement Flags:")
        for flag in result.disagreement_flags:
            lines.append(f"   • {flag}")

    return "\n".join(lines)


def save_result(result, output_dir: str = "output"):
    """Save debate result to JSON file."""
    Path(output_dir).mkdir(exist_ok=True)

    if isinstance(result, DualModelResult):
        data = {
            "scenario_id": result.scenario_id,
            "agreement": result.agreement,
            "disagreement_flags": result.disagreement_flags,
            "model_a": {
                "model": result.model_a_result.model,
                "decision": result.model_a_result.decision,
                "decision_label": result.model_a_result.decision_label,
                "probability": result.model_a_result.probability,
                "confidence": result.model_a_result.confidence,
                "reasoning": result.model_a_result.reasoning,
                "ceo_synthesis": result.model_a_result.ceo_synthesis,
                "agent_arguments": result.model_a_result.agent_arguments,
                "debate_log": [
                    {
                        "role_id": m.role_id,
                        "role_title": m.role_title,
                        "round": m.round_num,
                        "content": m.content,
                        "timestamp": m.timestamp,
                    }
                    for m in result.model_a_result.debate_log
                ],
                "token_usage": result.model_a_result.token_usage,
            },
            "model_b": {
                "model": result.model_b_result.model,
                "decision": result.model_b_result.decision,
                "decision_label": result.model_b_result.decision_label,
                "probability": result.model_b_result.probability,
                "confidence": result.model_b_result.confidence,
                "reasoning": result.model_b_result.reasoning,
                "ceo_synthesis": result.model_b_result.ceo_synthesis,
                "agent_arguments": result.model_b_result.agent_arguments,
                "debate_log": [
                    {
                        "role_id": m.role_id,
                        "role_title": m.role_title,
                        "round": m.round_num,
                        "content": m.content,
                        "timestamp": m.timestamp,
                    }
                    for m in result.model_b_result.debate_log
                ],
                "token_usage": result.model_b_result.token_usage,
            },
        }
        filename = f"{result.scenario_id}_dual.json"
    else:
        data = {
            "scenario_id": result.scenario_id,
            "model": result.model,
            "decision": result.decision,
            "decision_label": result.decision_label,
            "probability": result.probability,
            "confidence": result.confidence,
            "reasoning": result.reasoning,
            "ceo_synthesis": result.ceo_synthesis,
            "agent_arguments": result.agent_arguments,
            "debate_log": [
                {
                    "role_id": m.role_id,
                    "role_title": m.role_title,
                    "round": m.round_num,
                    "content": m.content,
                    "timestamp": m.timestamp,
                }
                for m in result.debate_log
            ],
            "token_usage": result.token_usage,
        }
        filename = f"{result.scenario_id}_{result.model}.json"

    filepath = Path(output_dir) / filename
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)

    print(f"\n💾 Results saved to {filepath}")
    return str(filepath)
