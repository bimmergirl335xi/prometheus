# attention.py
from infrastructure import CognitiveNode

class AttentionNode(CognitiveNode):
    """
    Top-Down Control.
    Suppresses signals that don't match the current 'Goal Context'.
    """
    def __init__(self, workspace):
        super().__init__("Attention", workspace, sensitivity=0.1)
        self.current_focus_topic = None

    def _calculate_relevance(self, packet: dict) -> float:
        # It monitors everything to filter it
        return 1.0 

    async def process(self, packet: dict):
        # If this is a GOAL setting packet, update our filter
        if packet.get('type') == "GOAL":
            self.current_focus_topic = packet['content']
            return None

        # If we have a focus, and this packet is unrelated noise...
        if self.current_focus_topic and packet.get('type') == "OBSERVATION":
            # Simple keyword matching for prototype
            if self.current_focus_topic not in str(packet['content']):
                # SUPPRESS: We can't actually delete from workspace, 
                # but we can post an INHIBIT signal or just not propagate it.
                # In this architecture, the node consumes it by doing nothing.
                pass
        return None