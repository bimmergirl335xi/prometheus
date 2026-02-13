# nodes.py
from infrastructure import CognitiveNode
from utils import log, Term, log_engine_event
from typing import Dict, Any
import asyncio
import random

from core_systems import PrincipleSystem

from engines.abstraction import AbstractionEngine
from engines.reasoning import ReasoningEngine
from engines.curiosity import CuriosityEngine
from engines.language import LanguageEngine
from engines.attention import AttentionNode

class ReasoningNode(CognitiveNode):
    def __init__(self, workspace, reasoning_engine):
        super().__init__("Reasoning", workspace, sensitivity=0.6)
        self.engine = reasoning_engine

    def _calculate_relevance(self, packet: Dict) -> float:

        if packet is None: return 0.0

        content = str(packet['content'])
        msg_type = packet.get('type', '')
        
        # High relevance if we see a question or conflict
        if "?" in content: return 0.9
        if msg_type == "CONFLICT": return 0.95
        if msg_type == "GOAL": return 0.7
        return 0.1

    async def process(self, packet: Dict) -> Dict:
        data = packet['content']
        
        # 1. TRACE INPUT
        log_engine_event("Reasoning", "Thinking...", data, "Processing")

        # 2. RUN LOGIC (Try to use the real engine if available)
        result_content = ""
        try:
            # Attempt to use the real engine logic if it exists
            if hasattr(self.engine, "validate_proposition"):
                valid, analysis = self.engine.validate_proposition(str(data))
                result_content = f"Analysis: {analysis} (Valid: {valid})"
            else:
                # Fallback to simulation if engine is missing
                result_content = f"Simulated deduction on: {data}"
        except Exception as e:
             result_content = f"Logic Error: {e}"

        # 3. CONSTRUCT PACKET
        response_packet = {
            "type": "ANSWER", 
            "content": result_content,
            "priority": 7.0
        }

        # 4. TRACE OUTPUT
        log_engine_event("Reasoning", "DECISION", data, response_packet)
        
        return response_packet

class AbstractionNode(CognitiveNode):
    def __init__(self, workspace, abstraction_engine):
        super().__init__("Abstraction", workspace, sensitivity=0.5)
        self.engine = abstraction_engine

    def _calculate_relevance(self, packet: Dict) -> float:

        if packet is None: return 0.0

        # Abstraction runs when we see concrete data or repeated patterns
        if packet.get('type') == "OBSERVATION": return 0.8
        return 0.2

    async def process(self, packet: Dict) -> Dict:
        data = packet['content']
        
        # 1. TRACE INPUT
        log_engine_event("Abstraction", "Pattern Matching", data, "Scanning...")

        # 2. RUN LOGIC
        # (Assuming your engine has a find_rule or similar method)
        found_pattern = None
        if hasattr(self.engine, "find_rule"):
             found_pattern = self.engine.find_rule(str(data))
        
        # 3. TRACE & RETURN
        if found_pattern:
            log_engine_event("Abstraction", "MATCH FOUND", data, found_pattern)
            return {"type": "THOUGHT", "content": f"Pattern detected: {found_pattern}", "priority": 6.0}
        
        # Log failure to find pattern (useful to know it tried)
        log_engine_event("Abstraction", "No Match", data, "None")
        return None
    
class CuriosityNode(CognitiveNode):
    """
    Wraps the CuriosityEngine.
    Monitors workspace traffic to update internal boredom/novelty models.
    Injects GOALs when the system is under-stimulated (Bored) or over-stimulated (Surprised).
    """
    def __init__(self, workspace, curiosity_engine):
        # We set sensitivity low (0.2) so we don't clog the bus, 
        # but we check often to update our internal boredom model.
        super().__init__("Curiosity", workspace, sensitivity=0.2)
        self.engine = curiosity_engine

    def _calculate_relevance(self, packet: Dict) -> float:
        # If the workspace is empty, we check if we are bored enough to hallucinate a goal
        if packet is None:
            return 0.8 if self.engine.boredom_level > 0.7 else 0.0

        # We are highly interested in new inputs (User/World) to calculate Novelty
        msg_type = packet.get("type", "")
        if msg_type in ["OBSERVATION", "USER", "RESULT", "FATAL"]:
            return 0.7

        return 0.1

    async def process(self, packet: Dict) -> Dict:
        # 1. Extract Context
        content = packet.get("content") if packet else {}
        
        # Ensure context is a dictionary for the engine to analyze
        context = content if isinstance(content, dict) else {"raw": str(content)}

        # 2. Run the Engine (Synchronous Logic)
        # This updates the boredom counters and checks Wundt curve thresholds
        generated_goal = self.engine.process_stimuli(context)

        # 3. Act on Engine Output
        if generated_goal:
            # The engine decided we need to do something (Boredom or Surprise triggered)
            log_engine_event("Curiosity", "INTRINSIC DRIVE", f"Boredom: {self.engine.boredom_level:.2f}", generated_goal['content'])
            return generated_goal
        
        # If nothing generated, we simply updated our internal state and return nothing.
        return None
    
class MotorNode(CognitiveNode):
    """
    The interface to the Cognitive Bridge / Robot.
    Executes actions physically.
    """
    def __init__(self, workspace, bridge):
        super().__init__("Motor", workspace, sensitivity=0.9)
        self.bridge = bridge

    def _calculate_relevance(self, packet: Dict) -> float:

        if packet is None: return 0.0

        # Only cares about ACTION packets
        if packet.get('type') == "ACTION": return 1.0
        return 0.0

    async def process(self, packet: Dict) -> Dict:
        cmd = packet['content'] # e.g. "MOVE_FORWARD"
        # self.bridge.dispatch_motor_command(cmd)
        return {"type": "FEEDBACK", "content": f"Executed: {cmd}"}

class RecursiveThoughtNode(CognitiveNode):
    """
    Creates a bounded internal recursion stream:
    THOUGHT -> THOUGHT (depth+1) until it converges or hits budget.
    """
    def __init__(self, workspace, reasoning_engine, max_depth: int = 5):
        super().__init__("Recursion", workspace, sensitivity=0.4)
        self.engine = reasoning_engine
        self.max_depth = int(max_depth)

    def _calculate_relevance(self, packet: Dict) -> float:
        if packet is None: return 0.0
        
        content = packet.get("content")
        packet_type = packet.get("type")
        
        # 1. Determine inner type safely
        inner_type = packet_type # Default
        
        if isinstance(content, dict):
            inner_type = content.get("type", packet_type)
        elif hasattr(content, "action_type"): 
            # It is an Action dataclass (Planner output)
            inner_type = "ACTION"

        # 2. Logic: React to Goals, Thoughts, or Actions
        if inner_type in ("GOAL", "THOUGHT", "ACTION"):
            return 0.9
            
        return 0.05
    
    async def process(self, packet: Dict) -> Dict:
        # 1. Safely extract content
        content = packet.get("content")
        
        # 2. Safely extract depth (Handle Action objects vs Dicts vs Strings)
        depth = 0
        if isinstance(content, dict):
            depth = int(content.get("depth", 0))
        elif hasattr(content, "metadata"):
            # Action objects store depth in their metadata
            depth = int(content.metadata.get("depth", 0))
        
        # 3. Check Recursion Limit
        if depth >= self.max_depth:
            return None # Stop thinking about this chain

        # 4. Generate next thought
        text_content = str(content)
        
        # Avoid infinite loops of "Elaborate: Elaborate: Elaborate..."
        if text_content.startswith("Elaborate:"):
            return None

        # Simple logic: If it's a question, answer it. Otherwise, elaborate.
        if "?" in text_content:
            return {
                "type": "THOUGHT",
                "content": f"Hypothesis for '{text_content}': It depends on context.",
                "depth": depth + 1,
                "priority": 6.0
            }
        
        return {
            "type": "THOUGHT", 
            "content": f"Elaborate: {text_content}", 
            "depth": depth + 1, 
            "priority": 4.0
        }
    
class AttentionNode(CognitiveNode):
    """Filters workspace content to maintain focus."""
    def __init__(self, workspace):
        super().__init__("Attention", workspace, sensitivity=0.3)
    
    def _calculate_relevance(self, packet: Dict) -> float:

        if packet is None: return 0.0

        # High relevance for alerts or new sensory data
        type_ = packet.get("type", "")
        if type_ in ["ALERT", "PAIN", "LOUD_NOISE"]: return 1.0
        return 0.1

    async def process(self, packet: Dict) -> Dict:
        # Just logging or boosting priority
        return None
    
class TheoryOfMindNode(CognitiveNode):
    """Models user intent."""
    def __init__(self, workspace):
        super().__init__("ToM", workspace, sensitivity=0.5)

    def _calculate_relevance(self, packet: Dict) -> float:

        if packet is None: return 0.0

        if packet.get("source") == "USER": return 0.9
        return 0.0

    async def process(self, packet: Dict) -> Dict:
        # Simple intent labeling
        return {
            "type": "THOUGHT",
            "content": "User intent appears to be information seeking.",
            "priority": 3.0
        }