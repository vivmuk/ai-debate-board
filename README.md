# AI Debate Board

Multi-agent AI debate orchestration framework powered by Venice.ai APIs.

Inspired by Liang Chang's biopharma decision experiment — where 5 AI agents role-playing
executives outperformed single-model predictions by 20-30% through structured adversarial debate.

## How It Works

```
┌─────────────────────────────────────────────┐
│  Scenario YAML (time-locked evidence)       │
│  ├── Roles with mandates                    │
│  ├── Options with IDs                       │
│  ├── Constitutional constraints             │
│  └── Pre-decision-date evidence only        │
└──────────────────┬──────────────────────────┘
                   │
    ┌──────────────▼──────────────┐
    │   PARALLEL DEBATE ENGINE    │
    │  ThreadPoolExecutor(5)      │
    │                             │
    │  Round 1: All agents ──►┐  │
    │  Round 2: All agents ──►┼──│──► See prior round
    │  Round N: All agents ──►┘  │    arguments
    └──────────────┬──────────────┘
                   │
    ┌──────────────▼──────────────┐
    │     CEO SYNTHESIS            │
    │  • Acknowledges all args    │
    │  • Overrules with reasoning │
    │  • Commits to ONE option    │
    │  • Calibrated probability    │
    └──────────────┬──────────────┘
                   │
    ┌──────────────▼──────────────┐
    │  DUAL-MODEL VALIDATION       │
    │  Model A ───┐                │
    │  Model B ───┼──► Compare    │
    │             │    decisions   │
    │             └──► Flag       │
    │                  divergence  │
    └─────────────────────────────┘
```

## Key Techniques

1. **Adversarial Role-Play** — each agent has a mandate forcing them to argue from their specific incentive structure
2. **Steel-Man Debate** — agents must restate the strongest opposing argument before rebutting
3. **Time-Locked Context** — constitutional constraints prevent post-decision-date information leakage
4. **Calibrated Probability** — CEO must commit to a numeric probability (0-100%)
5. **Dual-Model Cross-Validation** — run on two models independently; agreement = signal, divergence = flag
6. **Prediction Tracking** — forward-looking predictions get Brier scores and calibration curves

## Installation

```bash
pip install requests pyyaml rich
```

## Usage

```bash
# Run a debate on a scenario
python3 main.py -s scenarios/bms_checkmate026.yaml

# Dual-model cross-validation
python3 main.py -s scenarios/bms_checkmate026.yaml --dual-model

# Record a forward-looking prediction (tracks in ledger)
python3 main.py --predict -s scenarios/fomc_june_2026.yaml --dual-model

# Score a prediction after the outcome is known
python3 main.py --score PREDICTION_ID --outcome hold

# View prediction ledger
python3 main.py --ledger
```

## Scenarios

| Scenario | Type | Description |
|----------|------|-------------|
| `bms_checkmate026` | Retrospective | BMS Nivolumab 1L NSCLC: enrichment strategy (2012) |
| `merck_keynote024` | Retrospective | Merck Pembrolizumab 1L NSCLC: enrichment strategy (2014) |
| `fomc_june_2026` | **Forward-looking** | Will the Fed cut rates in June 2026? |

## Creating Custom Scenarios

Scenarios are YAML files. Key fields:

```yaml
scenario_id: my-prediction
title: "Will X happen by Y date?"
decision_date: "2026-06-01"
prediction_deadline: "2026-06-01"  # When the answer will be known
category: finance
forward_looking: true  # Enables prediction tracking

context: |
  Background information available at decision time...

options:
  - id: yes
    label: "Yes"
    description: "X will happen"
  - id: no
    label: "No"
    description: "X will not happen"

constitutional_constraints:
  - "You may NOT reference events after decision_date"

evidence:
  - type: data_source
    source: "Name"
    date: "2026-05-15"
    content: |
      Facts available at decision time...

roles:
  - id: ceo
    title: "Chief Decision Maker"
    mandate: "You make the FINAL call..."
    synthesis_role: true
  - id: bull
    title: "Bull Case Analyst"
    mandate: "Make the strongest case for YES..."
  - id: bear
    title: "Bear Case Analyst"
    mandate: "Make the strongest case for NO..."
  - id: contrarian
    title: "Contrarian"
    mandate: "Find the non-consensus view..."

debate_config:
  rounds: 2
  steel_man_required: true
```

## Prediction Calibration

The system tracks calibration over time using **Brier scores**:

- **Brier = 0**: Perfect prediction (100% confidence on a correct answer)
- **Brier = 0.25**: No-skill baseline (always predict 50%)
- **Brier = 1**: Perfectly wrong (100% confidence on the wrong answer)

After 10+ scored predictions, the calibration curve shows whether the system's
probabilities are well-calibrated (predicted 70% → actual ~70% correct).

## Architecture

- `venice_client.py` — Venice.ai API client with retry, fallback, and auth
- `debate_engine.py` — Core debate logic: prompts, rounds, CEO synthesis
- `parallel_engine.py` — ThreadPoolExecutor-based parallel agent execution
- `prediction_tracker.py` — Forward-looking prediction ledger with Brier scoring
- `main.py` — CLI entry point
- `scenarios/` — YAML scenario definitions

## Results (BMS CheckMate-026)

5 agents, 2 rounds, GLM-5.1:

| Agent | Round 1 | Round 2 | Shift |
|-------|---------|---------|-------|
| R&D | Adaptive/Stratified | Adaptive/Stratified | 🔒 |
| Commercial | All-Comers | All-Comers | 🔒 Ideological lock |
| Regulatory | Adaptive/Stratified | Adaptive/Stratified | 🔒 |
| CI | PD-L1 ≥1% | PD-L1 ≥1% | 🎯 Enrichment |
| Contrarian | PD-L1 ≥5% | PD-L1 ≥5% | 🎯 Sharpest |

**CEO Verdict**: Adaptive/Stratified @ 62% probability, medium confidence

The debate correctly identified biomarker enrichment as critical but chose the
safe option (adaptive/stratified) over the bold enrichment play (PD-L1 ≥5%).
Historically, the correct answer was PD-L1 ≥5% — matching the Contrarian's position.
