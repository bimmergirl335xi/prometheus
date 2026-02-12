# cognitive_bridge.py
import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from config import *
from utils import log, Term
from definitions import ActionType, Action, MetaEvent

# Import the engines type hints only (to avoid circular imports at runtime)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from abstraction import AbstractionEngine
    from reasoning import ReasoningEngine
    from language import LanguageEngine
    from robot_interface import RobotInterface
    from memory import LongTermMemory
    from health import CognitiveImmuneSystem
    from agents import MetaCognition

class CognitiveBridge:
    """
    The 'Brainstem' and 'Corpus Callosum'.
    1. Mediates data flow between Reason, Abstraction, and Language.
    2. Grounds abstract symbols in robotic sensor data.
    3. Modulates global plasticity based on metabolic/cognitive state.
    """
    def __init__(self, 
                 memory: 'LongTermMemory', 
                 robot: 'RobotInterface',
                 health: 'CognitiveImmuneSystem',
                 meta: 'MetaCognition'):
        self.memory = memory
        self.robot = robot
        self.health = health
        self.meta = meta
        
        # Engines will register themselves here
        self.abstraction_engine: Optional['AbstractionEngine'] = None
        self.reasoning_engine: Optional['ReasoningEngine'] = None
        self.language_engine: Optional['LanguageEngine'] = None
        
        # Bridge State
        self.global_plasticity = 1.0
        self.alertness = 1.0
        self.grounding_confidence = 0.5

    def register_engines(self, abstract, reason, lang):
        """Link the high-level cognitive engines."""
        self.abstraction_engine = abstract
        self.reasoning_engine = reason
        self.language_engine = lang

    def synchronize(self):
        """
        The 'Heartbeat'. 
        Called every cycle to sync Body (Robot) with Mind (Memory).
        """
        # 1. READ BODY: Get sensor state
        sensor_data = self.robot.get_context()
        
        # 2. GROUNDING: Update semantic memory with physical reality
        # If the robot sees "OBSTACLE_CLOSE", we stimulate that concept in memory
        if "spatial_concept" in sensor_data:
            concept = sensor_data["spatial_concept"]
            self.memory.semantic.stimulate(concept, strength=0.5)
            
            # Grounding check: Do we know what this means?
            if concept not in self.memory.semantic.concept_to_idx:
                log("BRIDGE", f"⚠️ Unfamiliar sensation: {concept}", Term.YELLOW)
                # Trigger a learning reflex?
        
        # 3. METABOLISM: Adjust alertness based on battery
        battery = sensor_data.get("battery", 100.0)
        self.alertness = max(0.1, min(1.0, battery / 50.0))
        
        # 4. MODULATION: Calculate plasticity for this tick
        self._calculate_plasticity()

    def _calculate_plasticity(self):
        """
        Decides how 'malleable' the brain is right now.
        High Surprise + High Health = High Plasticity.
        Low Energy or Sickness = Low Plasticity.
        """
        base_rate = 1.0
        
        # Factor 1: Surprise (Prediction Error)
        # (Assuming Executive pushes this to Meta, or we read it from Meta)
        # For now, we use Meta frustration as a proxy for "Needs Change"
        if self.meta.frustration > 0.6:
            base_rate *= 1.5 # Desperation learning
            
        # Factor 2: Health
        if sum(1 for m in self.health.health_metrics.values() if m.critical) > 0:
            base_rate *= 0.1 # Shut down learning to preserve integrity
            
        # Factor 3: Physical constraints
        if self.alertness < 0.3:
            base_rate *= 0.5 # Too tired to learn
            
        self.global_plasticity = base_rate

    def recursive_thought_loop(self, context: Dict[str, Any], depth: int = 0) -> Tuple[bool, Optional[Action]]:
        """
        The 'Inner Monologue' Loop.
        Orchestrates the ping-pong between Reason and Abstraction.
        """
        max_depth = 3
        log("BRIDGE", f"🔄 Loop Depth {depth}: Synthesizing...", Term.PURPLE)
        
        # 1. Ask Reason: "does this make sense?"
        valid, msg = self.reasoning_engine.validate_proposition(str(context))
        
        if not valid:
            # 2. If Logic fails, Ask Abstraction: "Do we have a rule for this anomaly?"
            log("BRIDGE", f"❌ Logic failed ({msg}). Querying Abstraction...", Term.ORANGE)
            found_rule = False
            for rule_name in self.abstraction_engine.abstractions:
                if self.abstraction_engine.apply_abstraction(rule_name, context):
                    log("BRIDGE", f"💡 Applied Heuristic: {rule_name}", Term.CYAN)
                    context["heuristic_applied"] = rule_name
                    found_rule = True
                    break
            
            if not found_rule:
                return False, None # Dead end
        
        # 3. Check for completion or continuation
        if depth >= max_depth:
            # We're done thinking, turn it into an action
            return True, Action(ActionType.SPEAK, "I have thought about this and concluded...", emergent=True)
            
        # 4. Continue the loop
        # Pass the refined context back for another pass
        return True, Action(
            ActionType.RECURSIVE_THINK, 
            payload={"depth": depth + 1, "parent_context": context},
            emergent=True
        )

    def dispatch_motor_command(self, command: str, payload: Any):
        """
        Safety layer between Mind and Body.
        Prevents dangerous commands.
        """
        # Safety Check 1: Is the body healthy?
        if self.alertness < 0.1:
            log("BRIDGE", "⛔ Battery too low for motion.", Term.RED)
            return
            
        # Safety Check 2: Obstacle override
        sensor_data = self.robot.get_context()
        if sensor_data.get("spatial_concept") == "OBSTACLE_CLOSE" and command == "FORWARD":
            log("BRIDGE", "⛔ REFLEX: Obstacle detected. Motion blocked.", Term.RED)
            return

        # Execute
        log("BRIDGE", f"⚡ Motor Command: {command}", Term.GREEN)
        self.robot.send_command(command, str(payload))