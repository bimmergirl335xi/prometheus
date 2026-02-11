# curiosity.py
from infrastructure import CognitiveNode
import random

class CuriosityNode(CognitiveNode):
    """
    Intrinsic Motivation.
    Fires when salience is low (boredom) or uncertainty is high.
    """
    def __init__(self, workspace):
        super().__init__("Curiosity", workspace, sensitivity=0.0) # Always watching
        self.boredom_counter = 0

    def _calculate_relevance(self, packet: dict) -> float:
        # If workspace is empty/null, relevance is HIGH
        if packet is None:
            self.boredom_counter += 1
            return 1.0 if self.boredom_counter > 50 else 0.0
        
        self.boredom_counter = 0
        return 0.0

    async def process(self, packet: dict):
        # We are bored. Generate a random research goal.
        topics = ["Entropy", "My own code", "The user", "Python optimization"]
        target = random.choice(topics)
        
        self.boredom_counter = 0
        return {
            "type": "GOAL", 
            "content": f"Investigate {target}"
        }