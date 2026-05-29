# AI Debate Board

Multi-agent AI debate orchestration framework powered by Venice.ai APIs.

Inspired by Liang Chang's biopharma decision experiment — where 5 AI agents role-playing
executives debated high-stakes strategic decisions with time-locked evidence, then a CEO
agent synthesized their arguments into a calibrated decision.

## Architecture

```
Scenario YAML → Agent Spawning → Multi-Round Debate → CEO Synthesis → Calibrated Decision
                     ↑                                     ↑
               Role Prompts                          Structured Output
               Time-Locked Context                   Probability Estimate
```

## Features

- **Multi-agent role-play**: Each agent has a defined mandate, incentives, and information boundaries
- **Time-locked evidence**: Constitutional constraints prevent "cheating" with future knowledge
- **Multi-round debate**: Agents argue, rebut, and revise across configurable rounds
- **Steel-man requirement**: Agents must accurately restate opposing positions before rebutting
- **CEO synthesis**: A final agent weighs all arguments and commits to a calibrated decision
- **Dual-model cross-validation**: Run the same scenario on multiple models — agreement = signal, divergence = escalation
- **Structured output**: Every decision comes with a probability estimate and confidence interval

## Quick Start

```bash
pip install -r requirements.txt

# Run a single scenario with default models
python main.py --scenario scenarios/bms_checkmate026.yaml

# Run with dual-model cross-validation
python main.py --scenario scenarios/bms_checkmate026.yaml --dual-model

# Run all scenarios in a directory
python main.py --scenarios-dir scenarios/ --dual-model
```

## Scenario Format

Scenarios are YAML files with:
- Decision context and date
- Available options (3-5 discrete choices)
- Time-locked evidence documents
- Agent role definitions
- Evaluation criteria

See `scenarios/` for examples.

## Debate Techniques

| Technique | Description |
|---|---|
| Adversarial Role-Play | Agents argue from their role's incentives (R&D vs Commercial vs Regulatory) |
| Steel-Man Debate | Must restate opposing position before rebutting |
| Devil's Advocate | Contrarian Board Member explicitly seeks asymmetric counter-positions |
| Delphi Convergence | Iterative rounds with anonymized summaries to reduce anchoring bias |
| Cross-Model Validation | Same framework on 2+ models; disagreements flagged for human review |

## Model Configuration

Uses Venice.ai API with configurable model cascade:
- Primary: `zai-org-glm-5-1`
- Secondary: `grok-4-20`
- Tertiary: `deepseek-v4-pro`

Set `VENICE_API_KEY` env var or it auto-reads from `~/.hermes/config.yaml`.
