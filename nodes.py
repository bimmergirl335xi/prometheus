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
        # Only cares about ACTION packets
        if packet.get('type') == "ACTION": return 1.0
        return 0.0

    async def process(self, packet: Dict) -> Dict:
        cmd = packet['content'] # e.g. "MOVE_FORWARD"
        # self.bridge.dispatch_motor_command(cmd)
        return {"type": "FEEDBACK", "content": f"Executed: {cmd}"}