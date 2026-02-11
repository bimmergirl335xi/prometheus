# health.py
import os
import time
import json
from collections import deque, Counter
from typing import Dict, List, Any

from config import HEALTH_FILE
from utils import log, Term
from definitions import HealthMetric, HealthStatus, Action, ActionType, FailureType
from core_systems import Genome

class CognitiveImmuneSystem:
    """
    Monitors cognitive health and triggers self-repair.
    Like a biological immune system, but for mental processes.
    """
    def __init__(self, genome: Genome):
        self.genome = genome
        self.health_metrics: Dict[HealthMetric, HealthStatus] = {
            metric: HealthStatus(metric, 0.5, 0.0)
            for metric in HealthMetric
        }
        self.anomaly_buffer = deque(maxlen=100)
        self.repair_history = deque(maxlen=50)
        self.last_checkup = time.time()
        self.load()

    def diagnose(self, state: Dict[str, Any]) -> Dict[HealthMetric, HealthStatus]:
        """Comprehensive health check based on agent state"""
        
        # COHERENCE
        if 'success_rate' in state and 'confidence' in state:
            alignment = 1.0 - abs(state['success_rate'] - state['confidence'])
            self._update_metric(HealthMetric.COHERENCE, alignment)
        
        # RESPONSIVENESS
        if 'avg_latency' in state:
            responsiveness = max(0.0, 1.0 - state['avg_latency'] / 5.0)
            self._update_metric(HealthMetric.RESPONSIVENESS, responsiveness)
        
        # ADAPTABILITY
        if 'concepts_learned_rate' in state:
            adaptability = min(1.0, state['concepts_learned_rate'] * 2.0)
            self._update_metric(HealthMetric.ADAPTABILITY, adaptability)
        
        # STABILITY
        if 'error_rate' in state:
            stability = 1.0 - state['error_rate']
            self._update_metric(HealthMetric.STABILITY, stability)
        
        # EFFICIENCY
        if 'memory_usage' in state and 'memory_capacity' in state:
            usage_ratio = state['memory_usage'] / state['memory_capacity']
            if 0.4 <= usage_ratio <= 0.7:
                efficiency = 1.0
            else:
                efficiency = 1.0 - abs(usage_ratio - 0.55) / 0.55
            self._update_metric(HealthMetric.EFFICIENCY, max(0.0, efficiency))
        
        # ROBUSTNESS
        if 'failure_recovery_rate' in state:
            self._update_metric(HealthMetric.ROBUSTNESS, state['failure_recovery_rate'])
        
        return self.health_metrics

    def _update_metric(self, metric: HealthMetric, new_value: float):
        status = self.health_metrics[metric]
        old_value = status.value
        
        # Exponential moving average
        alpha = 0.3
        status.value = alpha * new_value + (1 - alpha) * old_value
        status.trend = (status.value - old_value) * 10
        
        if status.value < 0.2 or (status.value < 0.4 and status.trend < -0.1):
            status.critical = True
            log("IMMUNE", f"⚠️  CRITICAL: {metric.name} at {status.value:.2f}", Term.RED)
        else:
            status.critical = False

    def prescribe_repair(self) -> List[Action]:
        """Generate repair actions based on critical health status."""
        repairs = []
        
        for metric, status in self.health_metrics.items():
            if status.critical:
                if metric == HealthMetric.COHERENCE:
                    repairs.append(Action(ActionType.REFLECT, "coherence_check"))
                    repairs.append(Action(ActionType.CONSOLIDATE, None))
                
                elif metric == HealthMetric.RESPONSIVENESS:
                    repairs.append(Action(ActionType.OPTIMIZE, "latency"))
                
                elif metric == HealthMetric.ADAPTABILITY:
                    current_rate = self.genome.traits.get('learning_rate', 0.1)
                    repairs.append(Action(
                        ActionType.SELF_MODIFY,
                        f"learning_rate={current_rate * 1.3}"
                    ))
                
                elif metric == HealthMetric.STABILITY:
                    current_risk = self.genome.traits.get('risk_tolerance', 0.3)
                    repairs.append(Action(
                        ActionType.SELF_MODIFY,
                        f"risk_tolerance={current_risk * 0.7}"
                    ))
                    repairs.append(Action(ActionType.CHECKPOINT, "stability_crisis"))
                
                elif metric == HealthMetric.EFFICIENCY:
                    repairs.append(Action(ActionType.PRUNE, "memory"))
                    repairs.append(Action(ActionType.CONSOLIDATE, None))
                
                elif metric == HealthMetric.ROBUSTNESS:
                    repairs.append(Action(ActionType.BUILD_SKILL, "error_recovery"))
        
        if repairs:
            self.repair_history.append((time.time(), [r.action_type.name for r in repairs]))
            log("IMMUNE", f"💊 Prescribed {len(repairs)} repair actions", Term.CYAN)
        
        return repairs

    def periodic_checkup(self, force: bool = False) -> bool:
        current_time = time.time()
        if not force and current_time - self.last_checkup < 60:
            return False
        
        self.last_checkup = current_time
        total_health = sum(s.value for s in self.health_metrics.values()) / len(self.health_metrics)
        critical_count = sum(1 for s in self.health_metrics.values() if s.critical)
        
        log("IMMUNE", f"🏥 Health: {total_health:.2f} | Critical: {critical_count}", 
            Term.CYAN if critical_count == 0 else Term.YELLOW)
        
        return critical_count > 0 or total_health < 0.4

    def save(self):
        try:
            data = {
                "metrics": {
                    m.name: {
                        "value": s.value,
                        "trend": s.trend,
                        "critical": s.critical
                    }
                    for m, s in self.health_metrics.items()
                },
                "repair_history": list(self.repair_history)
            }
            with open(HEALTH_FILE, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            log("IMMUNE", f"Save failed: {e}", Term.RED)

    def load(self):
        if os.path.exists(HEALTH_FILE):
            try:
                with open(HEALTH_FILE, 'r') as f:
                    data = json.load(f)
                for metric_name, metric_data in data.get("metrics", {}).items():
                    metric = HealthMetric[metric_name]
                    self.health_metrics[metric] = HealthStatus(
                        metric=metric,
                        value=metric_data["value"],
                        trend=metric_data["trend"],
                        critical=metric_data["critical"]
                    )
                self.repair_history = deque(data.get("repair_history", []), maxlen=50)
                log("IMMUNE", "Loaded health status", Term.CYAN)
            except Exception as e:
                log("IMMUNE", f"Load failed: {e}", Term.RED)