# nodes.py
from infrastructure import CognitiveNode
from typing import Dict, Any
import asyncio
import random

# Import your existing engines here
# from reasoning import ReasoningEngine
# from abstraction import AbstractionEngine

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
        # Use the existing engine logic
        # Assuming engine has a method 'deduce' or similar
        # For prototype, we map it:

        if "move" in str(data).lower():
            return {"type": "ACTION", "content": "MOVE_FORWARD 1", "priority": 8.0}
   
        if "?" in str(data):
            # Simulated deduction
            # valid, msg = self.engine.validate_proposition(data)
            return {"type": "ANSWER", "content": f"I reasoned about '{data}' -> It seems plausible."}
            
        return None

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
        # Check if this observation matches a known rule
        # matched = self.engine.find_rule(packet['content'])
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

        if packet is None:
            return 0.0
        t = packet.get("content", {}).get("type") if packet.get("type") == "WORKSPACE" else packet.get("type")
        # React to GOAL and THOUGHT packets
        if t in ("GOAL", "THOUGHT"):
            return 0.9
        return 0.05

    async def process(self, packet: Dict) -> Dict:
        inner = packet["content"] if packet.get("type") == "WORKSPACE" else packet
        depth = int(inner.get("depth", 0))

        if depth >= self.max_depth:
            return {"type": "NOTE", "content": "Recursion cap reached.", "priority": 2.0}

        content = inner.get("content", "")
        # Simple: if it looks like a question, let reasoning respond and then recurse once on the answer
        if "?" in str(content):
            ans = f"I reasoned about '{content}' -> It seems plausible."
            return {"type": "THOUGHT", "content": ans, "depth": depth + 1, "priority": 6.0}

        # Otherwise, shallow elaboration step
        return {"type": "THOUGHT", "content": f"Elaborate: {content}", "depth": depth + 1, "priority": 4.0}

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

class CuriosityNode(CognitiveNode):
    """Generates questions when uncertainty is high."""
    def __init__(self, workspace):
        super().__init__("Curiosity", workspace, sensitivity=0.4)
    
    def _calculate_relevance(self, packet: Dict) -> float:

        if packet is None: return 0.0

        # If content is "UNKNOWN" or low confidence, get curious
        content = str(packet.get("content", ""))
        if "unknown" in content.lower(): return 0.8
        return 0.1

    async def process(self, packet: Dict) -> Dict:
        content = packet.get("content", "")
        return {
            "type": "THOUGHT", 
            "content": f"What is the nature of {content}?", 
            "priority": 6.0
        }

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