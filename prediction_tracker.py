"""
AI Debate Board — Prediction Tracker

Tracks forward-looking predictions, their calibrated probabilities,
and scores them against actual outcomes over time. This is what turns
the debate board from a demo into a real forecasting tool.

Based on Philip Tetlock's superforecasting calibration principles:
- Brier scores for individual predictions
- Calibration curves across all predictions
- Discrimination (does the system distinguish likely from unlikely events?)
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict

from venice_client import VeniceConfig
from parallel_engine import ParallelDebateEngine
from debate_engine import load_scenario, save_result


# ─── Data structures ────────────────────────────────────────────────

@dataclass
class Prediction:
    """A single forward-looking prediction with calibration data."""
    prediction_id: str
    scenario_id: str
    question: str
    decision_date: str        # When the answer will be known
    created_at: str           # When this prediction was made
    
    # Debate outputs
    model_a: str = ""
    model_a_decision: str = ""
    model_a_probability: float = 0.0
    model_a_confidence: str = ""
    model_a_reasoning: str = ""
    
    model_b: str = ""
    model_b_decision: str = ""
    model_b_probability: float = 0.0
    model_b_confidence: str = ""
    model_b_reasoning: str = ""
    
    models_agree: bool = True
    
    # Scoring (filled when outcome is known)
    actual_outcome: Optional[str] = None
    scored_at: Optional[str] = None
    correct: Optional[bool] = None
    brier_score_a: Optional[float] = None
    brier_score_b: Optional[float] = None
    
    # File paths
    result_path_a: str = ""
    result_path_b: str = ""


class PredictionTracker:
    """Manages a ledger of predictions and their outcomes over time."""
    
    def __init__(self, ledger_path: str = "predictions_ledger.json"):
        self.ledger_path = Path(ledger_path)
        self.predictions: list[Prediction] = []
        self._load()
    
    def _load(self):
        """Load existing predictions from ledger."""
        if self.ledger_path.exists():
            with open(self.ledger_path) as f:
                data = json.load(f)
            self.predictions = [Prediction(**p) for p in data]
    
    def _save(self):
        """Save predictions to ledger."""
        with open(self.ledger_path, "w") as f:
            json.dump([asdict(p) for p in self.predictions], f, indent=2, default=str)
    
    def record_prediction(
        self,
        scenario: dict,
        result_a,
        result_b=None,
        result_path_a: str = "",
        result_path_b: str = "",
    ) -> Prediction:
        """Record a new prediction from debate results."""
        pred_id = f"{scenario['scenario_id']}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        pred = Prediction(
            prediction_id=pred_id,
            scenario_id=scenario["scenario_id"],
            question=scenario["title"],
            decision_date=scenario.get("prediction_deadline", scenario.get("decision_date", "")),
            created_at=datetime.utcnow().isoformat(),
            model_a=result_a.model,
            model_a_decision=result_a.decision,
            model_a_probability=result_a.probability,
            model_a_confidence=result_a.confidence,
            model_a_reasoning=result_a.reasoning[:500],
            model_b=result_b.model if result_b else "",
            model_b_decision=result_b.decision if result_b else "",
            model_b_probability=result_b.probability if result_b else 0,
            model_b_confidence=result_b.confidence if result_b else "",
            model_b_reasoning=result_b.reasoning[:500] if result_b else "",
            models_agree=(result_a.decision == result_b.decision) if result_b else True,
            result_path_a=result_path_a,
            result_path_b=result_path_b,
        )
        
        self.predictions.append(pred)
        self._save()
        return pred
    
    def score_prediction(
        self,
        prediction_id: str,
        actual_outcome: str,
    ) -> Prediction:
        """Score a prediction against the actual outcome."""
        pred = next((p for p in self.predictions if p.prediction_id == prediction_id), None)
        if not pred:
            raise ValueError(f"Prediction {prediction_id} not found")
        
        pred.actual_outcome = actual_outcome
        pred.scored_at = datetime.utcnow().isoformat()
        
        # Score model A
        pred.correct = (pred.model_a_decision == actual_outcome)
        pred.brier_score_a = self._brier_score(pred.model_a_probability, pred.correct)
        
        # Score model B if present
        if pred.model_b:
            correct_b = (pred.model_b_decision == actual_outcome)
            pred.brier_score_b = self._brier_score(pred.model_b_probability, correct_b)
        
        self._save()
        return pred
    
    @staticmethod
    def _brier_score(probability: float, correct: bool) -> float:
        """
        Brier score: (predicted_prob - actual_outcome)^2
        Lower is better. Range: 0 (perfect) to 1 (completely wrong).
        For probabilistic predictions, this penalizes both overconfidence and underconfidence.
        """
        actual = 1.0 if correct else 0.0
        predicted = probability / 100.0  # Convert from 0-100 to 0-1
        return (predicted - actual) ** 2
    
    def get_calibration_data(self) -> dict:
        """
        Compute calibration statistics across all scored predictions.
        
        A well-calibrated system should have:
        - Predictions at 70% confidence → ~70% should be correct
        - Predictions at 40% confidence → ~40% should be correct
        """
        scored = [p for p in self.predictions if p.actual_outcome is not None]
        if not scored:
            return {"scored_count": 0, "message": "No scored predictions yet"}
        
        # Bins: 0-20%, 20-40%, 40-60%, 60-80%, 80-100%
        bins = {f"{i}-{i+20}": {"total": 0, "correct": 0, "avg_prob": 0}
                for i in range(0, 100, 20)}
        
        total_brier_a = 0
        total_brier_b = 0
        b_count = 0
        
        for pred in scored:
            # Model A
            prob_bin = min(int(pred.model_a_probability / 20) * 20, 80)
            bin_key = f"{prob_bin}-{prob_bin+20}"
            bins[bin_key]["total"] += 1
            if pred.correct:
                bins[bin_key]["correct"] += 1
            bins[bin_key]["avg_prob"] += pred.model_a_probability
            total_brier_a += pred.brier_score_a or 0
            
            # Model B
            if pred.model_b and pred.brier_score_b is not None:
                total_brier_b += pred.brier_score_b
                b_count += 1
        
        # Compute hit rates
        calibration_curve = {}
        for bin_key, data in bins.items():
            if data["total"] > 0:
                avg_prob = data["avg_prob"] / data["total"]
                hit_rate = data["correct"] / data["total"]
                calibration_curve[bin_key] = {
                    "avg_predicted_prob": round(avg_prob, 1),
                    "actual_hit_rate": round(hit_rate * 100, 1),
                    "n": data["total"],
                    "calibration_gap": round(abs(avg_prob - hit_rate * 100), 1),
                }
        
        n_scored = len(scored)
        n_correct = sum(1 for p in scored if p.correct)
        
        return {
            "scored_count": n_scored,
            "correct_count": n_correct,
            "accuracy": round(n_correct / n_scored * 100, 1),
            "avg_brier_score_a": round(total_brier_a / n_scored, 3),
            "avg_brier_score_b": round(total_brier_b / b_count, 3) if b_count > 0 else None,
            "calibration_curve": calibration_curve,
            "dual_model_agreement_rate": sum(1 for p in scored if p.models_agree) / n_scored,
        }
    
    def list_pending(self) -> list[Prediction]:
        """List predictions that haven't been scored yet."""
        return [p for p in self.predictions if p.actual_outcome is None]
    
    def list_scored(self) -> list[Prediction]:
        """List predictions that have been scored."""
        return [p for p in self.predictions if p.actual_outcome is not None]
    
    def print_status(self):
        """Print a formatted status of all predictions."""
        print(f"\n{'='*60}")
        print(f"📊 PREDICTION LEDGER — {len(self.predictions)} total predictions")
        print(f"{'='*60}")
        
        pending = self.list_pending()
        scored = self.list_scored()
        
        if pending:
            print(f"\n⏳ Pending ({len(pending)}):")
            for p in pending:
                print(f"  • {p.question}")
                print(f"    ID: {p.prediction_id}")
                print(f"    Model A ({p.model_a}): {p.model_a_decision} @ {p.model_a_probability}%")
                if p.model_b:
                    print(f"    Model B ({p.model_b}): {p.model_b_decision} @ {p.model_b_probability}%")
                    agree = "✅ Agree" if p.models_agree else "⚠️ DISAGREE"
                    print(f"    Cross-validation: {agree}")
                print(f"    Decision date: {p.decision_date}")
                print()
        
        if scored:
            print(f"✅ Scored ({len(scored)}):")
            for p in scored:
                icon = "✅" if p.correct else "❌"
                print(f"  {icon} {p.question}")
                print(f"    Predicted: {p.model_a_decision} @ {p.model_a_probability}% | Actual: {p.actual_outcome}")
                print(f"    Brier: {p.brier_score_a:.3f}")
                print()
        
        # Calibration
        cal = self.get_calibration_data()
        if cal.get("scored_count", 0) > 0:
            print(f"📈 Calibration Summary:")
            print(f"   Accuracy: {cal['accuracy']}%")
            print(f"   Avg Brier Score (Model A): {cal['avg_brier_score_a']:.3f} (0=perfect, 1=worst)")
            if cal.get("avg_brier_score_b"):
                print(f"   Avg Brier Score (Model B): {cal['avg_brier_score_b']:.3f}")
            print(f"   Dual-model agreement rate: {cal['dual_model_agreement_rate']:.0%}")
            
            if cal.get("calibration_curve"):
                print(f"\n   Calibration Curve:")
                print(f"   {'Probability Bin':20} {'Predicted':>10} {'Actual':>10} {'Gap':>8} {'N':>4}")
                print(f"   {'─'*20} {'─'*10} {'─'*10} {'─'*8} {'─'*4}")
                for bin_key, data in cal["calibration_curve"].items():
                    print(f"   {bin_key:20} {data['avg_predicted_prob']:>9.1f}% "
                          f"{data['actual_hit_rate']:>9.1f}% "
                          f"{data['calibration_gap']:>7.1f}% "
                          f"{data['n']:>4}")


def run_and_record_prediction(
    scenario_path: str,
    dual_model: bool = True,
    output_dir: str = "output",
    ledger_path: str = "predictions_ledger.json",
):
    """Run a debate on a forward-looking scenario and record it as a prediction."""
    config = VeniceConfig.from_env_and_hermes()
    engine = ParallelDebateEngine(config, max_workers=5)
    tracker = PredictionTracker(ledger_path)
    
    scenario = load_scenario(scenario_path)
    
    if not scenario.get("forward_looking"):
        print(f"⚠️  Scenario {scenario['scenario_id']} is not marked as forward-looking.")
        print(f"   Add 'forward_looking: true' to the YAML to enable prediction tracking.")
    
    if dual_model:
        result = engine.run_dual_model_debate(scenario)
        from debate_engine import save_result
        save_result(result, output_dir)
        
        pred = tracker.record_prediction(
            scenario=scenario,
            result_a=result.model_a_result,
            result_b=result.model_b_result,
        )
    else:
        result_a = engine.run_debate(scenario)
        from debate_engine import save_result as sr
        path = sr(result_a, output_dir)
        
        pred = tracker.record_prediction(
            scenario=scenario,
            result_a=result_a,
            result_path_a=path,
        )
    
    print(f"\n📌 Prediction recorded: {pred.prediction_id}")
    print(f"   Question: {pred.question}")
    print(f"   Decision date: {pred.decision_date}")
    print(f"   Model A: {pred.model_a_decision} @ {pred.model_a_probability}%")
    if pred.model_b:
        print(f"   Model B: {pred.model_b_decision} @ {pred.model_b_probability}%")
    
    tracker.print_status()
    return pred


def score_existing_prediction(
    prediction_id: str,
    actual_outcome: str,
    ledger_path: str = "predictions_ledger.json",
):
    """Score a prediction against the actual outcome."""
    tracker = PredictionTracker(ledger_path)
    pred = tracker.score_prediction(prediction_id, actual_outcome)
    
    print(f"\n🏆 Prediction Scored: {pred.prediction_id}")
    print(f"   Predicted: {pred.model_a_decision} @ {pred.model_a_probability}%")
    print(f"   Actual: {actual_outcome}")
    icon = "✅ CORRECT" if pred.correct else "❌ WRONG"
    print(f"   Result: {icon}")
    print(f"   Brier Score: {pred.brier_score_a:.3f}")
    
    tracker.print_status()
    return pred
