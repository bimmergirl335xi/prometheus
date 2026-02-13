# social.py
from cognitive_infrastructure.infrastructure import CognitiveNode

class TheoryOfMindNode(CognitiveNode):
    """
    Social Simulation.
    Interprets 'USER_INPUT' not as text, but as 'Intent'.
    """
    def __init__(self, workspace):
        super().__init__("ToM", workspace, sensitivity=0.8)

    def _calculate_relevance(self, packet: dict) -> float:
        return 1.0 if packet.get('type') == "USER_INPUT" else 0.0

    async def process(self, packet: dict):
        text = packet['content']
        intent = "UNKNOWN"
        
        if "?" in text: intent = "INQUIRY"
        elif "!" in text: intent = "URGENCY"
        elif "bad" in text or "wrong" in text: intent = "CORRECTION"
        
        return {
            "type": "SOCIAL_INTENT",
            "content": f"User intends: {intent}"
        }